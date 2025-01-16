# VectorDBInt16.py
# Copyright© 2025 Constantine Vassilev. All rights reserved
import requests
import faiss
import numpy as np
from rocksdict import Rdict
import os
import logging
import json
from tqdm import tqdm
from typing import List, Dict

logger = logging.getLogger(__name__)

class VectorDBInt16:
    def __init__(self, folder, model="snowflake-arctic-l-v2.0", embedding_dim=1024, rdict_options=None):
        self.embedding_dim = embedding_dim
        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder
        self.float_embeddings = {}  # Store Float32 embeddings temporarily for comparison only

    def _setup_config(self, folder, model, embedding_dim):
        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            if os.path.exists(folder) and len(os.listdir(folder)) > 0:
                raise Exception(f"Folder {folder} contains files, but no config.json. If you want to create a new database, the folder must be empty.")
            os.makedirs(folder, exist_ok=True)
            with open(config_path, "w") as f:
                config = {'version': '1.0', 'model': model, 'embedding_dim': embedding_dim}
                json.dump(config, f)
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def _initialize_faiss_index(self, folder, embedding_dim):
        faiss_index_path = os.path.join(folder, "index.bin")
        if os.path.exists(faiss_index_path):
            index = faiss.read_index_binary(faiss_index_path)
            logger.info("Existing FAISS index loaded.")
        else:
            index = faiss.IndexBinaryIDMap2(faiss.IndexBinaryFlat(embedding_dim))
            logger.info(f"New FAISS index created with embedding dimension {embedding_dim}.")
        return index

    def _generate_embeddings(self, texts: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        url = "http://127.0.0.1:12345/v1/embeddings"
        results = {}
        for text in texts:
            try:
                response = requests.post(url, json={"model": self.config["model"], "input": text})
                response.raise_for_status()
                data = response.json()
                if 'data' in data and data['data']:
                    embedding = np.array(data['data'][0]['embedding'])
                    if embedding.shape[0] != self.embedding_dim:
                        logger.error(f"Unexpected embedding dimension: {embedding.shape[0]}. Expected: {self.embedding_dim}.")
                        continue

                    # Quantize embedding to Int16
                    quantized, min_val, max_val = self._quantize_to_int16(embedding)

                    results[text] = {
                        'float': embedding,  # Float32 embedding for temporary comparison
                        'int16': quantized,  # Int16 embedding for storage
                        'min_max': (min_val, max_val)  # Min-max values for dequantization
                    }
                else:
                    logger.warning(f"No embedding generated for text: {text}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: {text}. Error: {e}")
        return results

    @staticmethod
    def _quantize_to_int16(embedding: np.ndarray) -> (np.ndarray, float, float):
        min_val = np.min(embedding)
        max_val = np.max(embedding)
        if max_val == min_val:
            return np.zeros_like(embedding, dtype=np.int16), min_val, max_val
        scale = 32767 / max(abs(min_val), abs(max_val))
        return (embedding * scale).astype(np.int16), min_val, max_val

    @staticmethod
    def _dequantize_int16(emb_int16: np.ndarray, min_max: tuple) -> np.ndarray:
        min_val, max_val = min_max
        if max_val == min_val:
            return np.zeros_like(emb_int16, dtype=np.float32)
        scale = max(abs(min_val), abs(max_val)) / 32767
        return emb_int16.astype(np.float32) * scale

    def add_documents(self, doc_ids: List[int], docs: List[str], batch_size=64, save=True):
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        with tqdm(total=len(docs), desc="Indexing docs") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start:start + batch_size]
                batch_docs = docs[start:start + batch_size]
                embeddings = self._generate_embeddings(batch_docs)

                if not embeddings:
                    logger.error(f"Embedding generation failed for batch: {batch_docs}")
                    continue

                # Convert float32 embeddings to binary for FAISS IndexBinaryIDMap2
                binary_embeddings = np.array([
                    np.packbits((embeddings[doc]['float'] > 0).astype(np.uint8)) 
                    for doc in batch_docs
                ])

                # Add to FAISS index using binary embeddings
                self.index.add_with_ids(binary_embeddings, np.array(batch_ids, dtype=np.int64))

                for doc_id, doc in zip(batch_ids, batch_docs):
                    self.doc_db[str(doc_id)] = {
                        'doc': doc,
                        'emb_int16': embeddings[doc]['int16'],  # Store Int16 embedding
                        'min_max': embeddings[doc]['min_max']  # Store min-max values
                    }
                    # Float32 embeddings are stored temporarily, not in the database
                    self.float_embeddings[str(doc_id)] = embeddings[doc]['float']

                pbar.update(len(batch_docs))

        if save:
            self.save()

    def search(self, query: str, k=10, binary_oversample=10, compare_float32=False) -> List[Dict]:
        """
        Two-stage search mechanism:
        1. Approximate search using binary embeddings to reduce the candidate set.
        2. Refine scores with either float32 or int16 embeddings.
        """
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []

        query_embeddings = self._generate_embeddings([query])
        if not query_embeddings:
            logger.error("Query embedding generation failed. Returning empty results.")
            return []

        query_bin = np.packbits((query_embeddings[query]['float'] > 0).astype(np.uint8))
        query_float = query_embeddings[query]['float']

        # Phase I: Approximate search with binary embeddings
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_bin.reshape(1, -1), binary_k)
        initial_hits = [
            {'doc_id': doc_id, 'score_hamming': dist}
            for doc_id, dist in zip(ids[0], distances[0]) if doc_id != -1
        ]

        # Phase II: Refine scores using float32 or int16 embeddings
        hits = []
        for hit in initial_hits:
            doc_id_str = str(hit['doc_id'])
            doc_data = self.doc_db.get(doc_id_str)
            if doc_data is None:
                continue

            if compare_float32:
                doc_emb = self.float_embeddings.get(doc_id_str)
                if doc_emb is None:
                    logger.warning(f"Float32 embedding not available for doc_id={doc_id_str}. Skipping comparison.")
                    continue
            else:
                emb_int16 = doc_data['emb_int16']
                min_max = doc_data['min_max']
                doc_emb = self._dequantize_int16(emb_int16, min_max)

            score = float(np.dot(query_float, doc_emb))
            hits.append({
                "doc_id": hit['doc_id'],
                "score": score,
                "doc": doc_data['doc']
            })

        hits.sort(key=lambda x: x['score'], reverse=True)
        return hits[:k]

    def remove_document(self, doc_id: int, save=True):
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_db:
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            del self.doc_db[doc_id_str]
            del self.float_embeddings[doc_id_str]
            logger.info(f"Document {doc_id} removed.")
        else:
            logger.warning(f"Document {doc_id} not found in the database.")

        if save:
            self.save()

    def save(self):
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS index saved to disk.")

    def clear_float32_embeddings(self):
        """Clear Float32 embeddings from memory to save space."""
        self.float_embeddings.clear()
        logger.info("Cleared all Float32 embeddings from memory.")
