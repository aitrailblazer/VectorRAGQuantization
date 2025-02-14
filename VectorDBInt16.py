# VectorDBInt16.py
import os
import json
import logging
import numpy as np
import faiss
from rocksdict import Rdict
from typing import List, Dict
import requests

logger = logging.getLogger(__name__)

class VectorDBInt16:
    """
    A vector database class that uses ONLY int16 embeddings (no float).
    It calls a local embedding service (via embed_url) to retrieve 16-bit embeddings,
    then thresholds each dimension to 1 bit for storage in a FAISS Binary index.
    Document texts are stored in a RocksDict.

    Because FAISS's binary index stores exactly 1 bit per dimension, we lose 15 bits
    of precision from each 16-bit dimension. The search is Hamming distance only.
    """

    def __init__(
        self,
        folder: str,
        model: str = "snowflake-arctic-embed2",
        embedding_dim: int = 1024,
        rdict_options=None,
        embed_url: str = "http://localhost:11434/api/embed"
    ):
        """
        Args:
            folder: Directory path for the FAISS binary index & RocksDict data.
            model: Name/ID of the embedding model (sent to the local embed service).
            embedding_dim: Dimensionality of the embeddings (e.g., 1024).
            rdict_options: Optional RocksDict configuration.
            embed_url: Local embedding service endpoint (e.g., http://localhost:11434/api/embed).
        """
        self.embedding_dim = embedding_dim
        self.model = model
        self.embed_url = embed_url

        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        self.folder = folder

    def _setup_config(self, folder: str, model: str, embedding_dim: int):
        """
        Create or load a config.json to confirm we match the expected model + dim.
        """
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

        # Optional: validate the config matches our current settings
        if (self.config["model"] != model) or (self.config["embedding_dim"] != embedding_dim):
            logger.warning(
                "Config model/dim differs from constructor arguments. "
                f"config={self.config}, constructor=(model={model}, dim={embedding_dim})"
            )

    def _initialize_faiss_index(self, folder: str, embedding_dim: int):
        """
        Initialize or load a FAISS *binary* index. By thresholding each dimension of
        a 16-bit vector to 1 bit, we store 1 bit * embedding_dim total bits.
        """
        faiss_index_path = os.path.join(folder, "index.bin")
        if os.path.exists(faiss_index_path):
            index = faiss.read_index_binary(faiss_index_path)
            logger.info("Existing FAISS binary index loaded.")
        else:
            index = faiss.IndexBinaryIDMap2(faiss.IndexBinaryFlat(embedding_dim))
            logger.info(f"New FAISS binary index created with dimension {embedding_dim} (1 bit/dimension).")
        return index

    def _generate_int16_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Request int16 embeddings from the local embedding service at self.embed_url.

        The local service should accept a JSON payload with at least:
          {
            "model": <model_name>,
            "texts": [...],
            "embedding_bits": 16
          }
        and return something like:
          {
            "embeddings": [
              [ dim_0, dim_1, ..., dim_(embedding_dim-1) ],
              ...
            ]
          }

        This function returns a dict: text -> np.array(..., dtype=np.int16).
        """
        results = {}
        if not texts:
            return results

        payload = {
            "model": self.model,
            "texts": texts,
            "embedding_bits": 16
        }
        try:
            resp = requests.post(self.embed_url, json=payload)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Int16 embedding generation failed: {e}")
            return results

        resp_json = resp.json()
        embeddings = resp_json.get("embeddings", [])
        if len(embeddings) != len(texts):
            logger.error(
                f"Mismatch: got {len(embeddings)} embeddings for {len(texts)} texts."
            )
            return results

        # Convert each returned embedding to int16
        for i, text in enumerate(texts):
            arr = np.array(embeddings[i], dtype=np.int16)
            if arr.shape[0] != self.embedding_dim:
                logger.error(
                    f"Dimension mismatch for text={text}. "
                    f"Got {arr.shape[0]}, expected {self.embedding_dim}"
                )
                continue
            results[text] = arr
        return results

    @staticmethod
    def _to_binary(embedding: np.ndarray) -> np.ndarray:
        """
        Convert an int16 embedding to a 1-bit representation. We threshold each dimension
        around the mean dimension value. This collapses 16 bits => 1 bit per dimension.
        """
        threshold = np.mean(embedding)
        bit_mask = (embedding > threshold).astype(np.uint8)
        # Pack bits into bytes
        return np.packbits(bit_mask)

    def add_documents(
        self,
        doc_ids: List[int],
        docs: List[str],
        batch_size: int = 64,
        save: bool = True
    ):
        """
        Add documents in batches. Each doc is mapped to an int16 embedding, then
        thresholded to 1 bit/dimension, stored in FAISS for Hamming distance searches.
        The full text is stored in self.doc_db for retrieval.
        """
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length.")

        # Remove existing docs with same IDs to avoid duplicates
        for doc_id in doc_ids:
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        from tqdm import tqdm
        with tqdm(total=len(docs), desc="Indexing docs (Int16)") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start : start + batch_size]
                batch_texts = docs[start : start + batch_size]

                emb_map = self._generate_int16_embeddings(batch_texts)
                if not emb_map:
                    logger.error(f"Int16 embedding generation failed for batch: {batch_texts}")
                    pbar.update(len(batch_texts))
                    continue

                bin_embeddings = []
                valid_ids = []
                for doc_id_i, doc_text in zip(batch_ids, batch_texts):
                    emb = emb_map.get(doc_text)
                    if emb is None:
                        continue
                    ubin = self._to_binary(emb)
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
                            "int16": emb_map[doc_text]
                        }

                pbar.update(len(batch_texts))

        if save:
            self.save()

    def search(self, query: str, k: int = 10, binary_oversample: int = 10) -> List[Dict]:
        """
        Search for similar documents using purely 1-bit dimension embeddings (Hamming distance).
          1) Get int16 embedding for query
          2) Convert to 1-bit representation
          3) FAISS binary search
          4) Return top k (lowest Hamming distance)
        """
        if self.index.ntotal == 0:
            logger.error("No documents indexed. Please add documents before searching.")
            return []

        emb_map = self._generate_int16_embeddings([query])
        if not emb_map or query not in emb_map:
            logger.error("Query embedding generation failed; returning empty.")
            return []

        query_int16 = emb_map[query]
        query_bin = self._to_binary(query_int16)

        # Oversample to ensure enough candidates
        binary_k = min(k * binary_oversample, self.index.ntotal)
        distances, ids = self.index.search(query_bin.reshape(1, -1), binary_k)

        initial_hits = [
            (doc_id, dist)
            for doc_id, dist in zip(ids[0], distances[0])
            if doc_id != -1
        ]
        # Sort ascending by Hamming distance, then keep top k
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

    def remove_document(self, doc_id: int, save: bool = True):
        """
        Remove a document from both FAISS and RocksDict by doc_id.
        """
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_db:
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            del self.doc_db[doc_id_str]
            logger.info(f"Document {doc_id} removed.")
        else:
            logger.warning(f"Document {doc_id} not found.")
        if save:
            self.save()

    def save(self):
        """
        Persist the binary FAISS index to disk.
        """
        faiss_index_path = os.path.join(self.folder, "index.bin")
        faiss.write_index_binary(self.index, faiss_index_path)
        logger.info("FAISS int16 binary index saved to disk.")

    def __len__(self):
        """
        Return the number of documents in the index.
        """
        return self.index.ntotal


#
# Example helper functions that operate on the VectorDBInt16 class:
#
def find_closest_int16(vector_db: VectorDBInt16, query: str) -> Dict:
    """
    Return the single closest document by pure Hamming distance on the 1-bit embeddings.
    """
    results = vector_db.search(query, k=1)
    return results[0] if results else {}

def find_top_ten_int16(vector_db: VectorDBInt16, query: str) -> List[Dict]:
    """
    Return the top 10 matches by Hamming distance on the 1-bit embeddings.
    """
    results = vector_db.search(query, k=10)
    if results:
        print("Top 10 Results (Int16/Binary):")
        for r in results:
            print(f"Doc ID: {r['doc_id']}, Score: {r['score']}, Document: {r['doc']}")
            print("-" * 40)
    else:
        print("No matching documents found.")
    return results