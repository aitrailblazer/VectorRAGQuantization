# VectorDBInt8Global.py
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

class VectorDBInt8Global:
    """
    A vector database class using int8 quantization for embeddings, 
    with a single global clipping limit. All embeddings are clipped to ±global_limit 
    and scaled to -127..127.
    """

    def __init__(
        self,
        folder: str,
        model: str = "snowflake-arctic-embed2",
        embedding_dim: int = 1024,
        global_limit: float = 0.3,  # Example global limit ±0.3 for int8
        rdict_options=None,
        embed_url: str = "http://localhost:11434/api/embed"
    ):
        """
        Args:
            folder: Directory for storing FAISS index and doc DB.
            model: Name of the embedding model.
            embedding_dim: Dimensionality of each embedding.
            global_limit: Single global clipping limit (±global_limit).
            rdict_options: Optional dict config for RocksDict.
            embed_url: URL of the embedding service endpoint.
        """
        self.embedding_dim = embedding_dim
        self.global_limit = float(global_limit)
        self.embed_url = embed_url  # New optional parameter
        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder
        self.float_embeddings = {}  # For direct float32 comparisons

    def _setup_config(self, folder: str, model: str, embedding_dim: int):
        """
        Create or load a config.json. Store 'global_limit' for consistent usage across sessions.
        """
        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            if os.path.exists(folder) and len(os.listdir(folder)) > 0:
                raise Exception(
                    f"Folder {folder} contains files, but no config.json. "
                    f"If you want to create a new database, the folder must be empty."
                )
            os.makedirs(folder, exist_ok=True)
            with open(config_path, "w") as f:
                config = {
                    'version': '1.0',
                    'model': model,
                    'embedding_dim': embedding_dim,
                    'global_limit': self.global_limit
                }
                json.dump(config, f)
        with open(config_path, "r") as f:
            self.config = json.load(f)
        # Use the config's global_limit if it was loaded from disk
        self.global_limit = float(self.config.get("global_limit", self.global_limit))

    def _initialize_faiss_index(self, folder: str, embedding_dim: int):
        """
        Initialize or load a FAISS IndexBinaryIDMap2 for int8-based embeddings.
        """
        faiss_index_path = os.path.join(folder, "index.bin")
        if os.path.exists(faiss_index_path):
            index = faiss.read_index_binary(faiss_index_path)
            logger.info("Existing FAISS index loaded.")
        else:
            index = faiss.IndexBinaryIDMap2(faiss.IndexBinaryFlat(embedding_dim))
            logger.info(f"New FAISS index created with embedding dimension {embedding_dim}.")
        return index

    def _generate_embeddings(self, texts: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate float32 embeddings for each text, then quantize them to int8 via a global ±limit.
        """
        url = self.embed_url
        results = {}
        for text in texts:
            try:
                response = requests.post(url, json={"model": self.config["model"], "input": text})
                response.raise_for_status()
                data = response.json()
                if 'data' in data and data['data']:
                    embedding = np.array(data['data'][0]['embedding'], dtype=np.float32)
                elif 'embeddings' in data and data['embeddings']:
                    embedding = np.array(data['embeddings'], dtype=np.float32)
                else:
                    logger.warning(f"No embedding generated for text: {text}")
                    continue

                # If the embedding is wrapped in an extra dimension, squeeze it.
                if embedding.ndim > 1:
                    embedding = embedding[0]

                if embedding.shape[0] != self.embedding_dim:
                    logger.error(
                        f"Unexpected embedding dimension: {embedding.shape[0]}. "
                        f"Expected: {self.embedding_dim}. Skipping '{text}'."
                    )
                    continue

                # Clip and quantize to int8 using a single global limit
                q_int8 = self._quantize_to_int8(embedding, self.global_limit)

                results[text] = {
                    'float': embedding,  # For compare_float32
                    'ubinary': self._to_binary(embedding),
                    'int8': q_int8
                }
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: '{text}'. Error: {e}")
        return results

    @staticmethod
    def _quantize_to_int8(embedding: np.ndarray, limit: float) -> np.ndarray:
        """
        Global-limit int8 quantization:
          1. Clip to ±limit.
          2. scale = 127 / limit
          3. Round into [-127..127]
        """
        clipped = np.clip(embedding, -limit, limit)
        scale = 127.0 / limit
        scaled = np.round(clipped * scale)
        scaled = np.clip(scaled, -127, 127).astype(np.int8)
        return scaled

    @staticmethod
    def _dequantize_int8(emb_int8: np.ndarray, limit: float) -> np.ndarray:
        """
        Convert global-limit int8 back to float32:
          scale = limit / 127
          float_val = int8_val * scale
        """
        scale = limit / 127.0
        return emb_int8.astype(np.float32) * scale

    @staticmethod
    def _to_binary(embedding: np.ndarray) -> np.ndarray:
        """
        Create a simple binary array for FAISS approximate search,
        thresholding each dimension by the embedding mean.
        """
        return np.packbits((embedding > np.mean(embedding)).astype(np.uint8))

    def add_documents(self, doc_ids: List[int], docs: List[str], batch_size=64, save=True):
        """
        Add documents in batches, storing int8 embeddings with a single global limit.
        Also store float32 embeddings for direct comparison if desired.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        # Remove duplicates if any
        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        with tqdm(total=len(docs), desc="Indexing docs (Global Int8)") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start : start + batch_size]
                batch_docs = docs[start : start + batch_size]
                embeddings = self._generate_embeddings(batch_docs)

                if not embeddings:
                    logger.error(f"Embedding generation failed for batch: {batch_docs}")
                    continue

                # Use binary embeddings for approximate search in FAISS
                binary_embeddings = np.array(
                    [embeddings[doc]['ubinary'] for doc in batch_docs],
                    dtype=np.uint8
                )
                self.index.add_with_ids(binary_embeddings, np.array(batch_ids, dtype=np.int64))

                # Store data in RocksDict
                for doc_id_val, doc_text in zip(batch_ids, batch_docs):
                    self.doc_db[str(doc_id_val)] = {
                        'doc': doc_text,
                        'emb_int8': embeddings[doc_text]['int8']
                    }
                    self.float_embeddings[str(doc_id_val)] = embeddings[doc_text]['float']

                pbar.update(len(batch_docs))

        if save:
            self.save()

    def search(self, query: str, k=10, binary_oversample=10, compare_float32=False) -> List[Dict]:
        """
        Search with a FAISS binary index, then refine scores using either float32 or int8 embeddings 
        with a global limit.
        """
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []

        # Generate query embedding
        query_embeddings = self._generate_embeddings([query])
        if not query_embeddings:
            logger.error("Query embedding generation failed. Returning empty results.")
            return []

        query_bin = self._to_binary(query_embeddings[query]['float'])
        query_float = query_embeddings[query]['float']

        # Phase I: approximate candidate retrieval with binary embeddings
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_bin.reshape(1, -1), binary_k)
        initial_hits = [
            {'doc_id': doc_id, 'score_hamming': dist}
            for doc_id, dist in zip(ids[0], distances[0]) if doc_id != -1
        ]

        # Phase II: refine with dot product using float32 or dequantized int8 embeddings
        hits = []
        for hit in initial_hits:
            doc_id_str = str(hit['doc_id'])
            doc_data = self.doc_db.get(doc_id_str)
            if doc_data is None:
                continue

            if compare_float32:
                doc_emb = self.float_embeddings[doc_id_str]
            else:
                doc_emb = self._dequantize_int8(doc_data['emb_int8'], self.global_limit)

            score = float(np.dot(query_float, doc_emb))
            hits.append({
                "doc_id": hit['doc_id'],
                "score": score,
                "doc": doc_data['doc']
            })

        hits.sort(key=lambda x: x['score'], reverse=True)
        return hits[:k]

    def remove_document(self, doc_id: int, save=True):
        """
        Remove a document from FAISS index and RocksDict by doc_id.
        """
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
        """
        Save FAISS index to disk. RocksDict is consistent upon each write.
        """
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS index saved to disk.")

    def __len__(self):
        """
        Return how many vectors are currently stored in the FAISS index.
        """
        return self.index.ntotal