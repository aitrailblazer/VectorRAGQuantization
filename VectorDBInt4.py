# VectorDBInt4.py
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

class VectorDBInt4:
    """
    A vector database class using 4-bit (int4) quantization for embeddings,
    with per-document min/max scaling. Each document can have its own
    min/max range, ensuring flexible quantization at the cost of storing
    extra metadata and performing dequantization at query time.
    """

    def __init__(self, folder: str, model: str = "snowflake-arctic-embed2", embedding_dim: int = 1024, 
                 rdict_options=None, embed_url: str = "http://localhost:11434/api/embed"):
        """
        Initialize VectorDBInt4:

          folder: Directory path for saving index and docs.
          model: Name of the model used for embedding generation.
          embedding_dim: Expected dimensionality of embeddings.
          rdict_options: Optional configuration dict for RocksDict.
          embed_url: URL of the embedding service endpoint.
        """
        self.embedding_dim = embedding_dim
        self.embed_url = embed_url  # Configurable embedding endpoint
        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder
        self.float_embeddings = {}  # Store float32 embeddings for direct comparison

    def _setup_config(self, folder: str, model: str, embedding_dim: int):
        """
        Create or load a config.json, ensuring the folder is valid for a new or existing DB.
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

    def _initialize_faiss_index(self, folder: str, embedding_dim: int):
        """
        Initialize or load a binary FAISS index (IndexBinaryIDMap2) for approximate searching with int4.
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
        Generate float32 embeddings for each text from a local/remote service, then
        per-document quantize to int4 with min/max scaling.
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

                # If the embedding is wrapped in an extra dimension, extract the first element.
                if embedding.ndim > 1:
                    embedding = embedding[0]

                if embedding.shape[0] != self.embedding_dim:
                    logger.error(
                        f"Unexpected embedding dimension: {embedding.shape[0]}. "
                        f"Expected: {self.embedding_dim}. Skipping '{text}'."
                    )
                    continue

                # Quantize to int4 using per-document min/max scaling.
                q_packed, min_val, max_val = self._quantize_to_int4(embedding)
                results[text] = {
                    'float': embedding,
                    'ubinary': self._to_binary(embedding),
                    'int4': q_packed,
                    'min_max': (min_val, max_val),
                }
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: '{text}'. Error: {e}")
        return results

    @staticmethod
    def _quantize_to_int4(embedding: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Per-document 4-bit quantization using symmetrical min/max scaling:
          1. Compute min_val = min(embedding), max_val = max(embedding).
          2. Compute scale = 7 / max(abs(min_val), abs(max_val)).
          3. Round each dimension to an integer in the range [-8..7].
          4. Pack two 4-bit nibbles into one int8.
          
        Returns:
          (q_packed, min_val, max_val):
            - q_packed is the nibble-packed int4 array (approximately of shape (embedding_dim/2,)).
            - min_val, max_val are stored for dequantization.
        """
        min_val = float(np.min(embedding))
        max_val = float(np.max(embedding))
        if max_val == min_val:
            length_packed = (embedding.shape[0] + 1) // 2
            return np.zeros(length_packed, dtype=np.int8), min_val, max_val

        scale = 7.0 / max(abs(min_val), abs(max_val))
        scaled = np.round(embedding * scale)
        scaled = np.clip(scaled, -8, 7).astype(np.int8)

        length_packed = (scaled.shape[0] + 1) // 2
        q_packed = np.zeros(length_packed, dtype=np.int8)
        for i in range(length_packed):
            idxA = 2 * i
            idxB = 2 * i + 1
            valA = scaled[idxA] + 8  # shift range from [-8,7] to [0,15]
            if idxB < scaled.shape[0]:
                valB = scaled[idxB] + 8
            else:
                valB = 0
            combined = ((valA & 0x0F) << 4) | (valB & 0x0F)
            if combined > 127:
                combined -= 256
            q_packed[i] = np.int8(combined)
        return q_packed, min_val, max_val

    @staticmethod
    def _dequantize_int4(q_packed: np.ndarray, length: int, min_max: Tuple[float, float]) -> np.ndarray:
        """
        Unpack nibble-packed int4 data from q_packed and dequantize it to float32 using stored min/max.
        
        Steps:
          1. Compute scale = max(abs(min_val), abs(max_val)) / 7.
          2. Each nibble in [0, 15] is shifted to [-8, 7].
          3. Multiply by the scale.
        
        Returns:
          A float32 array of shape (length,).
        """
        min_val, max_val = min_max
        out = np.zeros(length, dtype=np.float32)
        if max_val == min_val:
            return out
        scale = max(abs(min_val), abs(max_val)) / 7.0
        idx = 0
        for byte in q_packed:
            byte_u8 = byte if byte >= 0 else (byte + 256)
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
        Add documents in batches, storing int4-quantized embeddings with per-document min/max.
        Also, store the original float32 embeddings for direct comparison.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        with tqdm(total=len(docs), desc="Indexing docs (Int4)") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start : start + batch_size]
                batch_docs = docs[start : start + batch_size]
                embeddings = self._generate_embeddings(batch_docs)
                if not embeddings:
                    logger.error(f"Embedding generation failed for batch: {batch_docs}")
                    continue

                binary_embeddings = np.array(
                    [embeddings[doc]['ubinary'] for doc in batch_docs],
                    dtype=np.uint8
                )
                self.index.add_with_ids(binary_embeddings, np.array(batch_ids, dtype=np.int64))
                for doc_id, doc_text in zip(batch_ids, batch_docs):
                    self.doc_db[str(doc_id)] = {
                        'doc': doc_text,
                        'emb_int4': embeddings[doc_text]['int4'],
                        'min_max': embeddings[doc_text]['min_max']
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
            if doc_data is None:
                continue

            if compare_float32:
                doc_emb = self.float_embeddings[doc_id_str]
            else:
                # Dequantize int4 embedding back to float32.
                doc_emb = self._dequantize_int4(
                    doc_data['emb_int4'],
                    self.embedding_dim,
                    doc_data['min_max']
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
        Remove a document from both the FAISS index and the document DB by its doc_id.
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
        Persist the FAISS binary index to disk. The document DB is updated immediately.
        """
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS index saved to disk.")

    def __len__(self):
        """
        Return the total number of documents currently indexed in FAISS.
        """
        return self.index.ntotal