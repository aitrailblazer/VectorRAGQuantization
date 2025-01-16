# VectorDBInt8.py
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

class VectorDBInt8:
    """
    A vector database class using int8 quantization for embeddings.
    Each document's embedding is quantized individually to preserve precision.
    """

    def __init__(self, folder, model="snowflake-arctic-l-v2.0", embedding_dim=1024, rdict_options=None):
        """
        Initialize the VectorDBInt8 class.

        Args:
            folder: Directory path for saving index and docs
            model: The model name for embedding generation
            embedding_dim: Expected dimensionality of embeddings
            rdict_options: Optional configuration for RocksDict
        """
        self.embedding_dim = embedding_dim
        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder
        self.float_embeddings = {}  # Store float32 embeddings for direct comparison

    def _setup_config(self, folder, model, embedding_dim):
        """
        Set up or load an existing config.json in the folder.
        If the folder doesn't exist, create it and store basic config.
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
                config = {'version': '1.0', 'model': model, 'embedding_dim': embedding_dim}
                json.dump(config, f)
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def _initialize_faiss_index(self, folder, embedding_dim):
        """
        Initialize a binary FAISS index (IndexBinaryIDMap2) or load an existing one.
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
        Generate embeddings for each text using your local/remote embedding service.
        Then quantize to int8 and store min/max for dequantization.
        """
        url = "http://127.0.0.1:12345/v1/embeddings"
        results = {}
        for text in texts:
            try:
                response = requests.post(url, json={"model": self.config["model"], "input": text})
                response.raise_for_status()
                data = response.json()
                if 'data' in data and data['data']:
                    embedding = np.array(data['data'][0]['embedding'], dtype=np.float32)
                    if embedding.shape[0] != self.embedding_dim:
                        logger.error(f"Unexpected embedding dimension: {embedding.shape[0]}. "
                                     f"Expected: {self.embedding_dim}. Skipping '{text}'.")
                        continue

                    # Quantize to int8
                    quantized, min_val, max_val = self._quantize_to_int8(embedding)

                    results[text] = {
                        'float': embedding,
                        'ubinary': self._to_binary(embedding),
                        'int8': quantized,
                        'min_max': (min_val, max_val),
                    }
                else:
                    logger.warning(f"No embedding generated for text: {text}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: '{text}'. Error: {e}")
        return results

    @staticmethod
    def _quantize_to_int8(embedding: np.ndarray) -> (np.ndarray, float, float):
        """
        Convert float32 embedding to int8 via symmetric min-max scaling.
          - scale = 127 / max(abs(min_val), abs(max_val))
        Returns (quantized_embedding, min_val, max_val).
        """
        min_val = np.min(embedding)
        max_val = np.max(embedding)
        if max_val == min_val:
            return np.zeros_like(embedding, dtype=np.int8), min_val, max_val
        scale = 127 / max(abs(min_val), abs(max_val))
        return (embedding * scale).astype(np.int8), min_val, max_val

    @staticmethod
    def _dequantize_int8(emb_int8: np.ndarray, min_max: tuple) -> np.ndarray:
        """
        Convert int8 embedding back to float32 using stored min/max values.
          - scale = max(abs(min_val), abs(max_val)) / 127
        """
        min_val, max_val = min_max
        if max_val == min_val:
            return np.zeros_like(emb_int8, dtype=np.float32)
        scale = max(abs(min_val), abs(max_val)) / 127
        return emb_int8.astype(np.float32) * scale

    @staticmethod
    def _to_binary(embedding: np.ndarray) -> np.ndarray:
        """
        Convert a float32 embedding to a binary embedding using thresholding
        based on the mean value. Useful for a quick approximate search.
        """
        return np.packbits((embedding > np.mean(embedding)).astype(np.uint8))

    def add_documents(self, doc_ids: List[int], docs: List[str], batch_size=64, save=True):
        """
        Add documents to the database in batches, generating and storing
        int8-quantized embeddings. Also store float32 embeddings for
        direct comparison.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        # If a doc_id already exists, remove it first to avoid duplicates
        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        with tqdm(total=len(docs), desc="Indexing docs (Int8)") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start:start + batch_size]
                batch_docs = docs[start:start + batch_size]
                embeddings = self._generate_embeddings(batch_docs)

                if not embeddings:
                    logger.error(f"Embedding generation failed for batch: {batch_docs}")
                    continue

                # Use binary embeddings in FAISS
                binary_embeddings = np.array([embeddings[doc]['ubinary'] for doc in batch_docs], dtype=np.uint8)
                self.index.add_with_ids(binary_embeddings, np.array(batch_ids, dtype=np.int64))

                # Store data in RocksDB
                for doc_id, doc in zip(batch_ids, batch_docs):
                    self.doc_db[str(doc_id)] = {
                        'doc': doc,
                        'emb_int8': embeddings[doc]['int8'],
                        'min_max': embeddings[doc]['min_max']
                    }
                    # Keep float32 for direct comparison (optional)
                    self.float_embeddings[str(doc_id)] = embeddings[doc]['float']

                pbar.update(len(batch_docs))

        if save:
            self.save()

    def search(self, query: str, k=10, binary_oversample=10, compare_float32=False) -> List[Dict]:
        """
        Search for similar documents based on an int8 or float32 comparison.

        Args:
            query: The query text.
            k: Number of top results to return.
            binary_oversample: How many candidates to retrieve from the binary index
                               before refining by dot product.
            compare_float32: If True, use float32 embeddings for final scoring
                             else use int8 embeddings.

        Returns:
            A list of dicts with the format:
              {
                "doc_id": int,
                "score": float,
                "doc": str
              }
        """
        # If no documents are indexed, return empty
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []

        # Generate query embedding
        query_embeddings = self._generate_embeddings([query])
        if not query_embeddings:
            logger.error("Query embedding generation failed. Returning empty results.")
            return []

        query_bin = query_embeddings[query]['ubinary']
        query_float = query_embeddings[query]['float']

        # Phase I: Approximate search with binary embeddings
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_bin.reshape(1, -1), binary_k)
        # Combine doc IDs with binary distances (Hamming)
        initial_hits = [
            {'doc_id': doc_id, 'score_hamming': dist}
            for doc_id, dist in zip(ids[0], distances[0]) if doc_id != -1
        ]

        # Phase II: Re-score with dot product of either float32 or int8
        hits = []
        for hit in initial_hits:
            doc_id = str(hit['doc_id'])
            doc_data = self.doc_db.get(doc_id)
            if not doc_data:
                continue

            if compare_float32:
                doc_emb = self.float_embeddings[doc_id]
            else:
                doc_emb = self._dequantize_int8(doc_data['emb_int8'], doc_data['min_max'])

            score = np.dot(query_float, doc_emb)
            hits.append({
                "doc_id": hit['doc_id'],
                "score": float(score),
                "doc": doc_data['doc']
            })

        # Sort final hits by score and keep top k
        hits.sort(key=lambda x: x['score'], reverse=True)
        return hits[:k]

    def remove_document(self, doc_id: int, save=True):
        """
        Remove a document from both the FAISS index and RocksDB by its doc_id.
        """
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_db:
            # FAISS binary index supports removal by ID
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            del self.doc_db[doc_id_str]
            del self.float_embeddings[doc_id_str]
            logger.info(f"Document {doc_id} removed from the database.")
        else:
            logger.warning(f"Document {doc_id} not found in the database.")

        if save:
            self.save()

    def save(self):
        """
        Persist the FAISS index to disk. The RocksDB is immediately consistent
        with every write, so no extra operation is required for doc_db.
        """
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS index saved to disk.")

    def __len__(self):
        """
        Return the total number of indexed documents based on FAISS index count.
        """
        return self.index.ntotal