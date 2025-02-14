import os
import json
import time
import numpy as np
import faiss
import logging
from rocksdict import Rdict
from typing import List, Dict
import requests

logger = logging.getLogger(__name__)

class CohereVectorDBInt8:
    """
    A vector database class that uses ONLY Cohere's int8 embeddings (no float embeddings).
    This class calls a Cohere Embed model (deployed in Azure AI/ML or Cohere's endpoint)
    to retrieve int8 embeddings, stores them in a FAISS Binary index, and
    keeps document texts in RocksDict.
    """

    def __init__(self, folder: str, model: str = "embed-english-v3.0", embedding_dim: int = 1024,
                 rdict_options=None):
        """
        Initialize the CohereVectorDBInt8 instance.

        Args:
            folder: Directory path to store the FAISS binary index and document database.
            model: Name of the Cohere model used for embedding generation.
            embedding_dim: Expected dimensionality of the int8 embeddings.
            rdict_options: Optional configuration for RocksDict.
        """
        self.embedding_dim = embedding_dim

        # Retrieve the endpoint and API key from the environment.
        self.endpoint = os.environ.get("COHERE_EMBED_ENDPOINT")
        if not self.endpoint:
            raise Exception("COHERE_EMBED_ENDPOINT is not set in the environment.")
        # Ensure the endpoint includes '/v2/embed'
        if "/v2/embed" not in self.endpoint:
            self.endpoint = self.endpoint.rstrip("/") + "/v2/embed"

        self.api_key = os.environ.get("COHERE_EMBED_KEY")
        if not self.api_key:
            raise Exception("COHERE_EMBED_KEY is not set in the environment.")

        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder

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
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def _initialize_faiss_index(self, folder: str, embedding_dim: int):
        """
        Initialize or load a FAISS binary index (IndexBinaryIDMap2).
        For int8 embeddings of length 1024, the binary representation is 1024 bits → 128 bytes.
        """
        faiss_index_path = os.path.join(folder, "index.bin")
        if os.path.exists(faiss_index_path):
            index = faiss.read_index_binary(faiss_index_path)
            logger.info("Existing FAISS binary index loaded.")
        else:
            index = faiss.IndexBinaryIDMap2(faiss.IndexBinaryFlat(embedding_dim))
            logger.info(f"New FAISS binary index created with embedding dimension {embedding_dim}.")
        return index

    def _generate_int8_embeddings(self, texts: List[str], input_type: str) -> Dict[str, np.ndarray]:
        """
        Request ONLY int8 embeddings from the Cohere endpoint for a given list of texts.
        Returns a dictionary mapping text -> int8 embedding (as a NumPy array).
        """
        results = {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.config["model"],
            "texts": texts,
            "input_type": input_type,
            "truncate": "NONE",
            "embedding_types": ["int8"]
        }
        logger.debug(f"Requesting int8 embeddings from: {self.endpoint}")
        logger.debug(f"Payload (int8): {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            resp_int8 = response.json()
            logger.debug("int8 response: %s", resp_int8)
        except Exception as e:
            logger.error("Int8 embedding generation failed: %s", str(e))
            return results

        for i, text in enumerate(texts):
            try:
                emb_int8 = np.array(resp_int8["embeddings"]["int8"][i], dtype=np.int8)
                # if shape is (1, 1024), squeeze out dimension
                if emb_int8.ndim > 1:
                    emb_int8 = emb_int8[0]
                if emb_int8.shape[0] != self.embedding_dim:
                    logger.error(
                        f"Embedding dimension mismatch for text='{text}'. "
                        f"Got {emb_int8.shape[0]}, expected {self.embedding_dim}. Skipping."
                    )
                    continue
                results[text] = emb_int8
            except Exception as ex:
                logger.error(f"Error processing int8 embedding for text='{text}': {ex}")
        return results

    @staticmethod
    def _to_binary(embedding: np.ndarray) -> np.ndarray:
        """
        Convert an int8 embedding to a binary array using thresholding at the mean.
        """
        return np.packbits((embedding > np.mean(embedding)).astype(np.uint8))

    def add_documents(self, doc_ids: List[int], docs: List[str], batch_size: int = 64, save: bool = True):
        """
        Add documents in batches, using only int8 embeddings from Cohere.
        The binary representation is used for indexing in FAISS.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        # Remove existing docs w/ same IDs to avoid duplicates
        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        from tqdm import tqdm
        with tqdm(total=len(docs), desc="Indexing docs (Int8)") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start : start + batch_size]
                batch_texts = docs[start : start + batch_size]

                emb_map = self._generate_int8_embeddings(batch_texts, input_type="search_document")
                if not emb_map:
                    logger.error(f"Int8 embedding generation failed for batch: {batch_texts}")
                    pbar.update(len(batch_texts))
                    continue

                # Convert each doc's int8 embedding to a binary representation
                bin_embeddings = []
                valid_ids = []
                for doc_id_i, doc_text in zip(batch_ids, batch_texts):
                    if doc_text not in emb_map:
                        continue
                    emb_int8 = emb_map[doc_text]
                    ubin = self._to_binary(emb_int8)
                    bin_embeddings.append(ubin)
                    valid_ids.append(doc_id_i)

                if not bin_embeddings:
                    pbar.update(len(batch_texts))
                    continue

                bin_embeddings = np.array(bin_embeddings, dtype=np.uint8)
                self.index.add_with_ids(bin_embeddings, np.array(valid_ids, dtype=np.int64))

                # Store doc text in doc_db
                for doc_id_i, doc_text in zip(batch_ids, batch_texts):
                    if doc_text in emb_map:
                        self.doc_db[str(doc_id_i)] = {
                            "doc": doc_text,
                            "int8": emb_map[doc_text]
                        }
                pbar.update(len(batch_texts))

        if save:
            self.save()

    def search(self, query: str, k: int = 10, binary_oversample: int = 10) -> List[Dict]:
        """
        Search for similar documents using int8 embeddings only.
        1) Generate an int8 embedding for the query.
        2) Convert to binary using thresholding at the mean.
        3) Use FAISS binary search to retrieve candidates based on Hamming distance.
        4) Return top k (lowest Hamming distance).
        """
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []

        emb_map = self._generate_int8_embeddings([query], input_type="search_query")
        if not emb_map or query not in emb_map:
            logger.error("Query embedding generation failed. Returning empty results.")
            return []

        query_int8 = emb_map[query]
        query_bin = self._to_binary(query_int8)

        # Oversample in binary search to ensure we get enough candidates
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_bin.reshape(1, -1), binary_k)
        initial_hits = [
            (doc_id, dist)
            for doc_id, dist in zip(ids[0], distances[0])
            if doc_id != -1
        ]

        # For a pure binary approach, we can just return the lowest distances.
        # Sort ascending by Hamming distance, then slice top k.
        initial_hits.sort(key=lambda x: x[1])
        initial_hits = initial_hits[:k]

        results = []
        for doc_id, dist in initial_hits:
            doc_id_str = str(doc_id)
            doc_data = self.doc_db.get(doc_id_str, {})
            results.append({
                "doc_id": doc_id,
                "score": dist,  # Lower = closer
                "doc": doc_data.get("doc", "N/A")
            })
        return results

    def search_rerank_cohere(self, query: str, k: int = 10, binary_oversample: int = 10,
                            rerank_model: str = "rerank-english-v3.0") -> List[Dict]:
        """
        Combine FAISS binary search with Cohere rerank.
        1) Generate an int8 embedding for the query and convert to binary.
        2) Use FAISS binary search to quickly retrieve an oversampled candidate list.
        3) Extract the candidate document texts from doc_db.
        4) Call Cohere's rerank API (via HTTP POST) to re-order the candidate documents.
        5) Return the top k reranked documents.
        
        Args:
            query (str): The query text.
            k (int): Number of top documents to return.
            binary_oversample (int): Oversampling factor for FAISS binary search.
            rerank_model (str): The model to use for reranking (default "rerank-english-v3.0").
        
        Returns:
            List[Dict]: A list of result dictionaries, each with "doc_id", "score", and "doc".
        """
        # Step 0: Read rerank endpoint and key from environment.
        rerank_endpoint = os.environ.get("COHERE_RERANK_ENDPOINT")
        if not rerank_endpoint:
            logger.error("COHERE_RERANK_ENDPOINT not set in the environment.")
            return []
        if not rerank_endpoint.endswith("/v2/rerank"):
            rerank_endpoint = rerank_endpoint.rstrip("/") + "/v2/rerank"
        
        rerank_key = os.environ.get("COHERE_RERANK_KEY")
        if not rerank_key:
            logger.error("COHERE_RERANK_KEY not set in the environment.")
            return []
        
        # Step 1: Generate query int8 embedding and convert to binary.
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []
        emb_map = self._generate_int8_embeddings([query], input_type="search_query")
        if not emb_map or query not in emb_map:
            logger.error("Query embedding generation failed. Returning empty results.")
            return []
        query_int8 = emb_map[query]
        query_bin = self._to_binary(query_int8)
        
        # Step 2: Use FAISS binary search to get candidate documents.
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_bin.reshape(1, -1), binary_k)
        initial_hits = [(doc_id, dist) for doc_id, dist in zip(ids[0], distances[0]) if doc_id != -1]
        initial_hits.sort(key=lambda x: x[1])
        initial_hits = initial_hits[:k * binary_oversample]
        
        # Step 3: Build candidate list from doc_db.
        candidates = []
        candidate_ids = []
        for doc_id, _ in initial_hits:
            doc_id_str = str(doc_id)
            doc_data = self.doc_db.get(doc_id_str, {})
            if "doc" in doc_data:
                candidates.append(doc_data["doc"])
                candidate_ids.append(doc_id)
        
        if not candidates:
            logger.error("No candidate documents found for reranking.")
            return []
        
        # Step 4: Call Cohere's rerank API via HTTP POST.
        logger.info("Calling Cohere rerank API at %s", rerank_endpoint)
        headers = {
            "Authorization": f"Bearer {rerank_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": rerank_model,
            "query": query,
            "top_n": k,
            "documents": candidates
        }
        logger.info("Payload for rerank: %s", payload)
        try:
            response = requests.post(rerank_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            rerank_response = response.json()
            logger.info("Rerank API response: %s", rerank_response)
        except Exception as e:
            logger.error("Rerank API call failed: %s", str(e))
            return []
        
        results_list = rerank_response.get("results")
        if not results_list:
            logger.error("Rerank response missing 'results'.")
            return []
        
        # Step 5: Build final reranked result list.
        reranked_results = []
        for result in results_list:
            idx = result["index"]  # index into the candidate list
            reranked_results.append({
                "doc_id": candidate_ids[idx],
                "score": result["relevance_score"],  # Higher is better.
                "doc": candidates[idx]
            })
        
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results
    
    def remove_document(self, doc_id: int, save: bool = True):
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_db:
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            del self.doc_db[doc_id_str]
            logger.info(f"Document {doc_id} removed from DB.")
        else:
            logger.warning(f"Document {doc_id} not found in DB.")
        if save:
            self.save()

    def save(self):
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))
        logger.info("FAISS int8 binary index saved to disk.")

    def __len__(self):
        return self.index.ntotal


#
# Example helper functions that operate on the CohereVectorDBInt8 class:
#
def find_closest_int8(vector_db: CohereVectorDBInt8, query: str) -> Dict:
    """
    Given a CohereVectorDBInt8 instance and a query string, returns the single closest document
    by pure int8/binary (Hamming) distance.
    """
    results = vector_db.search(query, k=1)
    return results[0] if results else {}

def find_top_ten_cohere_int8(vector_db: CohereVectorDBInt8, query: str) -> List[Dict]:
    """
    Given a CohereVectorDBInt8 instance and a query string, returns the top 10 matches
    by int8/binary (Hamming) distance.
    """
    results = vector_db.search(query, k=10)
    if results:
        print("Top Ten Results (int8/Binary):")
        for r in results:
            print(f"Doc ID: {r['doc_id']}, Score: {r['score']}, Document: {r['doc']}")
            print("-"*40)
    else:
        print("No matching documents found.")
    return results