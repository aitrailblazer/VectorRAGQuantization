# VectorDBInt4Global.py
# Copyright© 2025 Constantine Vassilev. All rights reserved

import requests
import faiss
import numpy as np
from rocksdict import Rdict
import os
import logging
import json
from tqdm import tqdm
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class VectorDBInt4Global:
    """
    A vector database class using 4-bit quantization for embeddings, with a single global clipping limit.
    We pack two int4 values into a single int8 for storage.
    """

    def __init__(
        self,
        folder: str,
        model: str = "snowflake-arctic-embed2",
        embedding_dim: int = 1024,
        global_limit: float = 0.18,   # A single global clipping limit, e.g., ±0.18
        rdict_options=None,
        embed_url: str = "http://localhost:11434/api/embed"
    ):
        """
        Args:
            folder: Directory where FAISS index and docs are stored.
            model: Model name for embedding generation.
            embedding_dim: Dimensionality of each embedding.
            global_limit: Single global clipping limit for int4 quantization (±global_limit).
            rdict_options: Optional configuration for RocksDict.
            embed_url: URL of the embedding service endpoint.
        """
        self.embedding_dim = embedding_dim
        self.global_limit = float(global_limit)
        self.embed_url = embed_url  # New parameter for endpoint
        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder
        self.float_embeddings = {}  # Store float32 embeddings for direct comparison

    def _setup_config(self, folder: str, model: str, embedding_dim: int):
        """
        Create or load a config.json. We also store our global limit in config for reproducibility.
        """
        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            if os.path.exists(folder) and len(os.listdir(folder)) > 0:
                raise Exception(
                    f"Folder {folder} contains files, but no config.json. "
                    "If you want to create a new database, the folder must be empty."
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
        Initialize or load a FAISS binary index (IndexBinaryIDMap2) using embedding_dim bits.
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
        Generate float32 embeddings for each text, then quantize them to int4 using a single global limit.
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

                # If the embedding is 2D (e.g., shape (1, 1024)), extract the first element.
                if embedding.ndim > 1:
                    embedding = embedding[0]

                if embedding.shape[0] != self.embedding_dim:
                    logger.error(
                        f"Unexpected embedding dimension: {embedding.shape[0]}. "
                        f"Expected: {self.embedding_dim}. Skipping '{text}'."
                    )
                    continue

                # Clip and quantize to int4 using the single global limit.
                q_packed = self._quantize_to_int4(embedding, self.global_limit)

                results[text] = {
                    'float': embedding,         # Store float for compare_float32
                    'ubinary': self._to_binary(embedding),
                    'int4': q_packed
                }
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: '{text}'. Error: {e}")
        return results

    @staticmethod
    def _quantize_to_int4(embedding: np.ndarray, limit: float) -> np.ndarray:
        """
        Symmetric quantization around zero to 4 bits with a global ±limit.
        Steps:
          1. Clip to ±limit.
          2. scale = 7 / limit.
          3. Round each dimension to an integer in [-8..7].
          4. Pack two 4-bit values into a single int8 nibble.
          
        Returns:
          q_packed, a nibble-packed int4 array.
        """
        min_val = float(np.min(embedding))
        max_val = float(np.max(embedding))
        if max_val == min_val:
            length_packed = (embedding.shape[0] + 1) // 2
            return np.zeros(length_packed, dtype=np.int8)
        scale = 7.0 / max(abs(min_val), abs(max_val))
        scaled = np.round(embedding * scale)
        scaled = np.clip(scaled, -8, 7).astype(np.int8)
        length_packed = (scaled.shape[0] + 1) // 2
        q_packed = np.zeros(length_packed, dtype=np.int8)
        for i in range(length_packed):
            idxA = 2 * i
            idxB = 2 * i + 1
            valA = scaled[idxA] + 8  # shift from [-8,7] to [0,15]
            if idxB < scaled.shape[0]:
                valB = scaled[idxB] + 8
            else:
                valB = 0
            combined = ((valA & 0x0F) << 4) | (valB & 0x0F)
            if combined > 127:
                combined -= 256
            q_packed[i] = np.int8(combined)
        return q_packed

    @staticmethod
    def _dequantize_int4(q_packed: np.ndarray, length: int, limit: float) -> np.ndarray:
        """
        Unpack nibble-packed int4 values and dequantize back to float32.
        Steps:
          1. Extract high and low nibbles from each byte.
          2. Shift values from [0,15] back to [-8,7].
          3. Multiply by scale = limit / 7.
        Returns a float32 array of shape (length,).
        """
        out = np.zeros(length, dtype=np.float32)
        scale = limit / 7.0
        idx = 0
        for byte in q_packed:
            byte_u8 = byte if byte >= 0 else byte + 256
            high_nibble = (byte_u8 >> 4) & 0x0F
            low_nibble = byte_u8 & 0x0F
            out[idx] = (high_nibble - 8) * scale
            idx += 1
            if idx < length:
                out[idx] = (low_nibble - 8) * scale
                idx += 1
        return out

    @staticmethod
    def _to_binary(embedding: np.ndarray) -> np.ndarray:
        """
        Convert a float32 embedding to a binary array for FAISS approximate search,
        thresholding each dimension by the embedding mean.
        """
        return np.packbits((embedding > np.mean(embedding)).astype(np.uint8))

    def add_documents(self, doc_ids: List[int], docs: List[str], batch_size: int = 64, save: bool = True):
        """
        Add documents in batches, storing int4-quantized embeddings (with a single global limit).
        Also store the float32 embeddings for direct comparison.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")
        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)
        with tqdm(total=len(docs), desc="Indexing docs (Global Int4)") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start : start + batch_size]
                batch_docs = docs[start : start + batch_size]
                embeddings = self._generate_embeddings(batch_docs)
                if not embeddings:
                    logger.error(f"Embedding generation failed for batch: {batch_docs}")
                    continue
                binary_embeddings = np.array(
                    [self._to_binary(embeddings[doc]['float']) for doc in batch_docs],
                    dtype=np.uint8
                )
                self.index.add_with_ids(binary_embeddings, np.array(batch_ids, dtype=np.int64))
                for doc_id, doc_text in zip(batch_ids, batch_docs):
                    self.doc_db[str(doc_id)] = {
                        'doc': doc_text,
                        'emb_int4': embeddings[doc_text]['int4']
                    }
                    self.float_embeddings[str(doc_id)] = embeddings[doc_text]['float']
                pbar.update(len(batch_docs))
        if save:
            self.save()

    def search(self, query: str, k: int = 10, binary_oversample: int = 10, compare_float32: bool = False) -> List[Dict]:
        """
        1. Retrieve initial candidates using FAISS binary search on nibble-packed embeddings.
        2. Refine scores using dot product on either the float32 embeddings or dequantized int4 embeddings.
        """
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []
        query_embeddings = self._generate_embeddings([query])
        if not query_embeddings:
            logger.error("Query embedding generation failed. Returning empty results.")
            return []
        query_bin = self._to_binary(query_embeddings[query]['float'])
        query_float = query_embeddings[query]['float']
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_bin.reshape(1, -1), binary_k)
        initial_hits = [
            {'doc_id': doc_id, 'score_hamming': dist}
            for doc_id, dist in zip(ids[0], distances[0]) if doc_id != -1
        ]
        hits = []
        for hit in initial_hits:
            doc_id_str = str(hit['doc_id'])
            doc_data = self.doc_db.get(doc_id_str)
            if not doc_data:
                continue
            if compare_float32:
                doc_emb = self.float_embeddings[doc_id_str]
            else:
                doc_emb = self._dequantize_int4(
                    doc_data['emb_int4'],
                    self.embedding_dim,
                    self.config.get("global_limit", 0.18)
                )
            score = float(np.dot(query_float, doc_emb))
            hits.append({
                "doc_id": hit['doc_id'],
                "score": score,
                "doc": doc_data['doc']
            })
        hits.sort(key=lambda x: x['score'], reverse=True)
        return hits[:k]

    def remove_document(self, doc_id: int, save: bool = True):
        """
        Remove a document from the FAISS index and doc DB by its doc_id.
        """
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_db:
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
        Save the FAISS index to disk. The doc DB is updated immediately.
        """
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS index saved to disk.")

    def __len__(self):
        """
        Return the total number of documents currently indexed in FAISS.
        """
        return self.index.ntotal