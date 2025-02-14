# CohereVectorDBBinary.py
# Copyright© 2025 Constantine Vassilev. All rights reserved
#
# This module implements a vector database class for Cohere embeddings using
# signed binary quantization. Each document’s float32 embedding is converted into
# a signed binary vector (values -1 or +1) by thresholding at the mean. For compact
# storage and fast approximate search, the signed binary vector is mapped to 0/1 and
# then packed into a binary array using np.packbits.
#
# At retrieval time, the packed binary is unpacked back to a signed binary vector,
# which can be used to compute similarity (e.g. via dot product) with a query’s float32 embedding.
#
# The embedding service is accessed using the Azure AI Inference SDK.
# The endpoint URL is read from the environment variable COHERE_EMBED_ENDPOINT
# (which should include the full URL, e.g. "https://AITCohere-embed-v3-english.eastus.models.ai.azure.com/v2/embed"),
# and the API key is read from AZURE_INFERENCE_CREDENTIAL.

import os
import json
import numpy as np
import faiss
from rocksdict import Rdict
import logging
from tqdm import tqdm
from typing import List, Dict
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

class CohereVectorDBBinary:
    """
    A vector database class using signed binary quantization for Cohere embeddings.
    Each document's float32 embedding is generated via the Cohere API (hosted on Azure) 
    and converted to a binary vector (values -1 or +1) using the mean as a threshold.
    The binary vector is then mapped to 0/1 and packed for compact storage.
    """

    def __init__(self, folder: str, model: str = "cohere-embed-english-v3.0", embedding_dim: int = 1024,
                 rdict_options=None):
        """
        Initialize the CohereVectorDBBinary class.

        Args:
            folder: Directory for saving the FAISS index and document database.
            model: Cohere model name for embedding generation.
            embedding_dim: Expected dimensionality of embeddings.
            rdict_options: Optional configuration for RocksDict.
        """
        self.embedding_dim = embedding_dim

        # Read the endpoint from the environment variable.
        endpoint = os.environ.get("COHERE_EMBED_ENDPOINT")
        if not endpoint:
            raise Exception("COHERE_EMBED_ENDPOINT is not set in the environment.")
        self.embed_url = endpoint  # Expect full endpoint URL (e.g. .../v2/embed)

        # Read API key from environment.
        api_key = os.environ.get("COHERE_EMBED_KEY")
        if not api_key:
            raise Exception("COHERE_EMBED_KEY is not set in the environment.")

        # Create the EmbeddingsClient using the Azure AI Inference SDK.
        self.client = EmbeddingsClient(
            endpoint=self.embed_url,
            credential=AzureKeyCredential(api_key)
        )

        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder
        self.float_embeddings = {}  # Store original float32 embeddings for optional scoring

    def _setup_config(self, folder: str, model: str, embedding_dim: int):
        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            if os.path.exists(folder) and os.listdir(folder):
                raise Exception(
                    f"Folder {folder} contains files, but no config.json. To create a new database, the folder must be empty."
                )
            os.makedirs(folder, exist_ok=True)
            with open(config_path, "w") as f:
                config = {"version": "1.0", "model": model, "embedding_dim": embedding_dim}
                json.dump(config, f)
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def _initialize_faiss_index(self, folder: str, embedding_dim: int):
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
        Generate float32 embeddings using the Cohere API via Azure Inference,
        then convert them to a signed binary representation.
        """
        results = {}
        for text in texts:
            try:
                # Prepare payload; the EmbeddingsClient expects a list of strings.
                payload = {"input": [text]}
                response = self.client.embed(payload)
                if not response.data:
                    logger.warning(f"No embedding returned for text: {text}")
                    continue
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                if embedding.ndim > 1:
                    embedding = embedding[0]
                if embedding.shape[0] != self.embedding_dim:
                    logger.error(f"Unexpected embedding dimension: {embedding.shape[0]}. Expected: {self.embedding_dim}. Skipping '{text}'.")
                    continue

                # Convert float32 embedding to signed binary vector (-1 or +1)
                signed_binary = self._to_signed_binary(embedding)
                # Pack the signed binary vector for compact storage.
                packed_binary = self._pack_signed_binary(signed_binary)

                results[text] = {
                    "float": embedding,
                    "packed_binary": packed_binary
                }
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: '{text}'. Error: {e}")
        return results

    @staticmethod
    def _to_signed_binary(embedding: np.ndarray) -> np.ndarray:
        """
        Convert a float32 embedding into a signed binary vector.
        Each element is set to +1 if it is greater than or equal to the mean,
        and -1 otherwise.
        """
        mean_val = np.mean(embedding)
        return np.where(embedding >= mean_val, 1, -1).astype(np.int8)

    @staticmethod
    def _pack_signed_binary(signed_binary: np.ndarray) -> np.ndarray:
        """
        Pack a signed binary vector (with values -1 and +1) into a compact binary representation.
        Maps -1 to 0 and +1 to 1, then uses np.packbits.
        """
        bits = ((signed_binary + 1) // 2).astype(np.uint8)
        return np.packbits(bits)

    @staticmethod
    def _unpack_signed_binary(packed: np.ndarray, length: int) -> np.ndarray:
        """
        Unpack a packed binary representation and convert back to a signed binary vector.
        Unpacked bits (0,1) are mapped back to (-1, +1).
        """
        bits = np.unpackbits(packed)[:length]
        return np.where(bits == 0, -1, 1).astype(np.float32)

    def add_documents(self, doc_ids: List[int], docs: List[str], batch_size: int = 64, save: bool = True):
        """
        Add documents in batches. For each document, generate a float32 embedding via Azure Inference,
        convert it to a signed binary vector, pack it, and store it in FAISS and RocksDict.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        for start in range(0, len(docs), batch_size):
            batch_ids = doc_ids[start:start + batch_size]
            batch_docs = docs[start:start + batch_size]
            embeddings = self._generate_embeddings(batch_docs)
            if not embeddings:
                logger.error(f"Embedding generation failed for batch: {batch_docs}")
                continue

            # For FAISS, add the packed binary representation.
            binary_embeddings = np.array(
                [embeddings[doc]["packed_binary"] for doc in batch_docs],
                dtype=np.uint8
            )
            self.index.add_with_ids(binary_embeddings, np.array(batch_ids, dtype=np.int64))
            for doc_id, doc in zip(batch_ids, batch_docs):
                self.doc_db[str(doc_id)] = {
                    "doc": doc,
                    "packed_binary": embeddings[doc]["packed_binary"]
                }
                self.float_embeddings[str(doc_id)] = embeddings[doc]["float"]
        if save:
            self.save()

    def search(self, query: str, k: int = 10, binary_oversample: int = 10, compare_float32: bool = False) -> List[Dict]:
        """
        Search for similar documents using a two-stage approach:
          1. Retrieve candidates using FAISS binary search on the packed binary representation.
          2. Rescore candidates using the dot product between the query float32 embedding and either
             the original float32 embedding or the unpacked signed binary vector.
        """
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []

        query_embeddings = self._generate_embeddings([query])
        if not query_embeddings:
            logger.error("Query embedding generation failed. Returning empty results.")
            return []

        query_float = query_embeddings[query]["float"]
        query_packed = self._pack_signed_binary(self._to_signed_binary(query_float))

        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_packed.reshape(1, -1), binary_k)
        initial_hits = [
            {"doc_id": doc_id, "score_hamming": dist}
            for doc_id, dist in zip(ids[0], distances[0]) if doc_id != -1
        ]

        hits = []
        for hit in initial_hits:
            doc_id_str = str(hit["doc_id"])
            doc_data = self.doc_db.get(doc_id_str)
            if not doc_data:
                continue
            if compare_float32:
                doc_emb = self.float_embeddings[doc_id_str]
            else:
                doc_emb = self._unpack_signed_binary(doc_data["packed_binary"], self.embedding_dim)
            score = float(np.dot(query_float, doc_emb))
            hits.append({
                "doc_id": hit["doc_id"],
                "score": score,
                "doc": doc_data["doc"]
            })
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:k]

    def remove_document(self, doc_id: int, save: bool = True):
        """
        Remove a document from both the FAISS index and the document database by its doc_id.
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
        Save the FAISS index to disk. The document database is maintained consistently.
        """
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS index saved to disk.")

    def __len__(self):
        """
        Return the total number of indexed documents based on the FAISS index.
        """
        return self.index.ntotal