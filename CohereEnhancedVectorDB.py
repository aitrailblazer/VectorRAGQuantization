#!/usr/bin/env python
"""
CohereEnhancedVectorDB.py

An enhanced Cohere-based vector database that uses HTTP requests to the Cohere embed endpoint.
It leverages multiple embedding types:
  - int8: Stored for final cosine similarity rescoring.
  - ubinary: Used for fast FAISS binary indexing and Hamming distance search.
  - float: Used for dot-product and cosine similarity rescoring.

Environment Variables:
  - COHERE_EMBED_ENDPOINT: The endpoint for Cohere embedding API (e.g., https://api.cohere.ai/v2/embed)
  - COHERE_EMBED_KEY: The API key for accessing the Cohere embedding API.
"""

import os
import json
import time
import numpy as np
import faiss
import logging
import requests
import tqdm
from rocksdict import Rdict
from typing import List, Dict

logger = logging.getLogger(__name__)


class CohereEnhancedVectorDB:
    """
    Enhanced Cohere-based vector database that uses HTTP requests to the Cohere embedding endpoint.
    
    This class:
      - Uses COHERE_EMBED_ENDPOINT and COHERE_EMBED_KEY to generate embeddings.
      - Obtains multiple embedding types: int8, ubinary, and float.
      - Converts int8 embeddings into binary (ubinary) representations for FAISS indexing.
      - Stores the document text along with its int8 embedding in RocksDict.
      - Provides a three-phase search:
          Phase I: Fast FAISS binary search using the ubinary representation.
          Phase II: Rescoring via dot-product between the query's float embedding and unpacked binary document embeddings.
          Phase III: Final rescoring using cosine similarity between the query's float embedding and the stored int8 embeddings.
    """

    def __init__(self,
                 folder: str,
                 model: str = "embed-english-v3.0",
                 embedding_dim: int = 1024,
                 index_type=faiss.IndexBinaryFlat,
                 index_args: List = None,
                 rdict_options=None):
        """
        Initialize the CohereEnhancedVectorDB.

        Args:
            folder (str): Directory to store the FAISS index and document DB.
            model (str): Cohere model to use.
            embedding_dim (int): Dimension of the embeddings.
            index_type: FAISS index type to use (default: faiss.IndexBinaryFlat).
            index_args (list): Additional arguments to pass to the FAISS index constructor.
            rdict_options: Options for RocksDict.
        """
        if index_args is None:
            index_args = [embedding_dim]

        # Set up endpoint and API key from environment variables.
        self.endpoint = os.environ.get("COHERE_EMBED_ENDPOINT")
        if not self.endpoint:
            raise Exception("COHERE_EMBED_ENDPOINT is not set in the environment.")
        if "/v2/embed" not in self.endpoint:
            self.endpoint = self.endpoint.rstrip("/") + "/v2/embed"

        self.api_key = os.environ.get("COHERE_EMBED_KEY")
        if not self.api_key:
            raise Exception("COHERE_EMBED_KEY is not set in the environment.")

        self.embedding_dim = embedding_dim
        self.model = model
        self.folder = folder

        # Setup folder and config file.
        self._setup_config(folder, model, embedding_dim)

        # Initialize FAISS binary index.
        self.index = self._initialize_faiss_index(folder, embedding_dim, index_type, index_args)

        # Initialize RocksDict for document storage.
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)

    def _setup_config(self, folder: str, model: str, embedding_dim: int):
        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            if os.path.exists(folder) and os.listdir(folder):
                raise Exception(
                    f"Folder {folder} contains files, but no config.json. "
                    "To create a new database, the folder must be empty."
                )
            os.makedirs(folder, exist_ok=True)
            with open(config_path, "w") as f:
                config = {
                    "version": "1.0",
                    "model": model,
                    "embedding_dim": embedding_dim
                }
                json.dump(config, f)
        else:
            with open(config_path, "r") as f:
                config = json.load(f)
            # If there is a mismatch, log a warning and update.
            if config.get("model") != model or config.get("embedding_dim") != embedding_dim:
                logger.warning("Config model or embedding_dim mismatch. Overwriting config.")
                config = {"version": "1.0", "model": model, "embedding_dim": embedding_dim}
                with open(config_path, "w") as fOut:
                    json.dump(config, fOut)
        self.config = config

    def _initialize_faiss_index(self, folder: str, embedding_dim: int, index_type, index_args: List):
        """
        Initialize or load a FAISS binary index.
        """
        faiss_index_path = os.path.join(folder, "index.bin")
        if os.path.exists(faiss_index_path):
            index = faiss.read_index_binary(faiss_index_path)
            logger.info("Existing FAISS binary index loaded.")
        else:
            index = faiss.IndexBinaryIDMap2(index_type(*index_args))
            logger.info(f"New FAISS binary index created with embedding dimension {embedding_dim}.")
        return index

    def _to_binary(self, emb_int8: np.ndarray) -> np.ndarray:
        """
        Convert an int8 embedding to a binary (ubinary) array using thresholding at the mean.
        """
        return np.packbits((emb_int8 > np.mean(emb_int8)).astype(np.uint8))

    def _get_embeddings(self, texts: List[str], input_type: str, embedding_types: List[str]) -> Dict:
        """
        Obtain embeddings by calling the Cohere embed endpoint via HTTP POST.

        Args:
            texts (List[str]): List of texts to embed.
            input_type (str): Either "search_document" or "search_query".
            embedding_types (List[str]): List of embedding types to request, e.g., ["int8", "ubinary", "float"].

        Returns:
            Dict: A dictionary of embeddings returned by the endpoint.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.config["model"],
            "texts": texts,
            "input_type": input_type,
            "truncate": "NONE",
            "embedding_types": embedding_types
        }
        logger.debug(f"Requesting {embedding_types} embeddings from: {self.endpoint}")
        logger.debug(f"Payload {embedding_types}: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("embeddings", {})
        except Exception as e:
            logger.error("Embedding generation failed: %s", str(e))
            return {}

    def add_documents(self,
                      doc_ids: List[int],
                      docs: List[str],
                      batch_size: int = 64,
                      save: bool = True):
        """
        Add documents in batches. For each document, obtain both int8 and ubinary embeddings.
        The FAISS index is updated using ubinary embeddings, and RocksDict stores the int8 embedding.

        Args:
            doc_ids (List[int]): List of unique document IDs.
            docs (List[str]): List of document texts.
            batch_size (int): Number of documents per batch.
            save (bool): Whether to save the FAISS index after adding.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        # Remove duplicates: if an ID already exists, remove it.
        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        with tqdm.tqdm(total=len(docs), desc="Indexing documents") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start:start+batch_size]
                batch_docs = docs[start:start+batch_size]

                emb = self._get_embeddings(batch_docs,
                                           input_type="search_document",
                                           embedding_types=["int8", "ubinary"])
                if not emb:
                    logger.error("Failed to retrieve embeddings for a batch.")
                    pbar.update(len(batch_docs))
                    continue

                try:
                    # Expecting emb["int8"] and emb["ubinary"] as lists.
                    int8_embs = np.array(emb["int8"], dtype=np.int8)
                    ubinary_embs = np.array(emb["ubinary"], dtype=np.uint8)
                except Exception as e:
                    logger.error("Error processing embeddings: %s", str(e))
                    pbar.update(len(batch_docs))
                    continue

                # Add ubinary embeddings to FAISS index.
                self.index.add_with_ids(ubinary_embs, np.array(batch_ids, dtype=np.int64))

                # Store document text along with its int8 embedding.
                for doc_id, doc, emb_int8 in zip(batch_ids, batch_docs, int8_embs):
                    self.doc_db[str(doc_id)] = {"doc": doc, "int8": emb_int8}
                pbar.update(len(batch_docs))

        if save:
            self.save()

    def search(self,
               query: str,
               k: int = 10,
               binary_oversample: int = 10,
               int8_oversample: int = 3) -> List[Dict]:
        """
        Search for similar documents using a three-phase approach:
          Phase I: Use FAISS binary search (ubinary) to retrieve an oversampled candidate set.
          Phase II: Rescore using dot-product between the query's float embedding and unpacked binary document embeddings.
          Phase III: Final rescoring using cosine similarity between the query's float embedding and the stored int8 document embeddings.

        Args:
            query (str): The search query.
            k (int): Number of top documents to return.
            binary_oversample (int): Factor to oversample candidates in the binary search.
            int8_oversample (int): Factor to oversample candidates for final rescoring.

        Returns:
            List[Dict]: A list of result dictionaries with keys "doc_id", "score", and "doc".
        """
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []

        # Phase I: Retrieve query embeddings.
        emb = self._get_embeddings([query],
                                   input_type="search_query",
                                   embedding_types=["float", "ubinary"])
        if not emb:
            logger.error("Query embedding generation failed.")
            return []

        try:
            query_float = np.array(emb["float"], dtype=np.float32)
            query_ubinary = np.array(emb["ubinary"], dtype=np.uint8)
        except Exception as e:
            logger.error("Error processing query embeddings: %s", str(e))
            return []

        # Use ubinary for fast binary search.
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_ubinary.reshape(1, -1), binary_k)
        initial_hits = [
            {"doc_id": int(doc_id), "score_hamming": int(dist)}
            for doc_id, dist in zip(ids[0], distances[0])
            if doc_id != -1
        ]
        initial_hits.sort(key=lambda x: x["score_hamming"])
        candidates = initial_hits[:k * binary_oversample]

        if not candidates:
            logger.error("No candidates found.")
            return []

        # Phase II: Rescore using dot-product between query_float and unpacked binary document embeddings.
        start_time = time.time()
        for hit in candidates:
            try:
                # Retrieve binary embedding from the index.
                doc_bin = self.index.reconstruct(hit["doc_id"])
                # Unpack bits (0/1) then map to -1/+1.
                doc_unpacked = np.unpackbits(doc_bin, axis=-1).astype(np.int32)
                doc_unpacked = 2 * doc_unpacked - 1
                hit["score_binary"] = float(query_float[0].dot(doc_unpacked))
            except Exception as e:
                logger.error("Error in phase II rescoring for doc_id %s: %s", hit["doc_id"], e)
                hit["score_binary"] = -np.inf
        logger.info("Phase II rescoring (binary dot-product) took %.2f ms", (time.time()-start_time)*1000)

        candidates.sort(key=lambda x: x["score_binary"], reverse=True)
        rescoring_candidates = candidates[:k * int8_oversample]

        # Phase III: Final rescoring using cosine similarity between query_float and stored int8 embeddings.
        start_time = time.time()
        final_results = []
        for hit in rescoring_candidates:
            doc_entry = self.doc_db.get(str(hit["doc_id"]))
            if not doc_entry:
                continue
            try:
                doc_int8 = np.array(doc_entry["int8"], dtype=np.int8)
                doc_norm = np.linalg.norm(doc_int8)
                if doc_norm == 0:
                    cos_sim = -np.inf
                else:
                    cos_sim = float(query_float[0].dot(doc_int8)) / doc_norm
                hit["score_cosine"] = cos_sim
                hit["doc"] = doc_entry.get("doc", "N/A")
                final_results.append(hit)
            except Exception as e:
                logger.error("Error in phase III rescoring for doc_id %s: %s", hit["doc_id"], e)
                continue
        logger.info("Phase III rescoring (cosine similarity) took %.2f ms", (time.time()-start_time)*1000)

        final_results.sort(key=lambda x: x["score_cosine"], reverse=True)
        return final_results[:k]

    def remove_document(self, doc_id: int, save: bool = True):
        """
        Remove a document from both the FAISS index and RocksDict.

        Args:
            doc_id (int): The document ID to remove.
            save (bool): Whether to save the FAISS index after removal.
        """
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_db:
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            del self.doc_db[doc_id_str]
            logger.info(f"Document {doc_id} removed.")
        else:
            logger.warning(f"Document {doc_id} not found in the database.")
        if save:
            self.save()

    def save(self):
        """
        Save the FAISS index to disk.
        """
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS binary index saved.")

    def __len__(self):
        return self.index.ntotal


# Example helper functions:

def find_closest_document(db: CohereEnhancedVectorDB, query: str) -> Dict:
    """
    Retrieve the single closest document for a given query.
    """
    results = db.search(query, k=1)
    return results[0] if results else {}


def print_top_results(db: CohereEnhancedVectorDB, query: str, k: int = 10):
    """
    Print the top k search results.
    """
    results = db.search(query, k=k)
    if results:
        print(f"Top {k} Results:")
        for res in results:
            print(f"Doc ID: {res['doc_id']}, Cosine Score: {res['score_cosine']:.4f}")
            print(f"Document: {res['doc']}")
            print("-" * 40)
    else:
        print("No matching documents found.")