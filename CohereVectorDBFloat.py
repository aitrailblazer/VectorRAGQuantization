# CohereVectorDBFloat.py
import os
import json
import logging
import numpy as np
import faiss
from rocksdict import Rdict
from typing import List, Dict
import requests

logger = logging.getLogger(__name__)

class CohereVectorDBFloat:
    """
    A vector DB class that requests *float* embeddings from Cohere (model=embed-english-v3.0),
    and stores them in a FAISS *float* index (e.g., IndexIDMap + IndexFlatIP).
    """

    def __init__(self, folder: str, model: str = "embed-english-v3.0", embedding_dim: int = 1024,
                 rdict_options=None):
        self.embedding_dim = embedding_dim
        self.folder = folder
        # Cohere environment variables
        self.endpoint = os.environ.get("COHERE_EMBED_ENDPOINT")
        if not self.endpoint:
            raise Exception("COHERE_EMBED_ENDPOINT not set.")
        if "/v2/embed" not in self.endpoint:
            self.endpoint = self.endpoint.rstrip("/") + "/v2/embed"
        self.api_key = os.environ.get("COHERE_EMBED_KEY")
        if not self.api_key:
            raise Exception("COHERE_EMBED_KEY not set.")

        self.model = model
        self._setup_config(folder, model, embedding_dim)
        self.index = self._initialize_faiss_index(folder, embedding_dim)
        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)

    def _setup_config(self, folder: str, model: str, embedding_dim: int):
        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            if os.path.exists(folder) and os.listdir(folder):
                raise Exception(
                    f"Folder {folder} not empty but no config.json found. "
                    "To create new DB, folder must be empty or have config.json."
                )
            os.makedirs(folder, exist_ok=True)
            with open(config_path, "w") as f:
                config = {"model": model, "embedding_dim": embedding_dim}
                json.dump(config, f)
        else:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            # optionally compare config vs. constructor

    def _initialize_faiss_index(self, folder: str, embedding_dim: int):
        faiss_index_path = os.path.join(folder, "index.faiss")
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            logger.info("Existing float FAISS index loaded.")
        else:
            # Dot-product or L2-based float index.  Example:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
            logger.info(f"New float FAISS index created (dim={embedding_dim}).")
        return index

    def _generate_float_embeddings(self, texts: List[str], input_type: str):
        """
        Calls Cohere with "embedding_types": ["float"] to get real float embeddings.
        Return: dict[text] = np.array(float32)
        """
        results = {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "texts": texts,
            "input_type": input_type,
            "truncate": "NONE",
            "embedding_types": ["float"]
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # data["embeddings"]["float"] -> list of lists
        except Exception as e:
            logger.error(f"Float embedding request failed: {e}")
            return results

        float_list = data["embeddings"]["float"]
        for i, txt in enumerate(texts):
            emb = np.array(float_list[i], dtype=np.float32)
            if emb.ndim > 1:
                emb = emb[0]
            if emb.shape[0] != self.embedding_dim:
                logger.warning(f"Dimension mismatch for text '{txt}': got {emb.shape[0]}, want {self.embedding_dim}")
                continue
            results[txt] = emb
        return results

    def add_documents(self, doc_ids: List[int], docs: List[str], batch_size: int = 64, save: bool = True):
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must match length.")

        from tqdm import tqdm
        for doc_id in doc_ids:
            # remove if already in doc_db
            if str(doc_id) in self.doc_db:
                self.remove_document(doc_id, save=False)

        with tqdm(total=len(docs), desc="Indexing docs (Float)") as pbar:
            for start in range(0, len(docs), batch_size):
                batch_ids = doc_ids[start: start+batch_size]
                batch_txts= docs[start: start+batch_size]
                emb_map = self._generate_float_embeddings(batch_txts, input_type="search_document")
                if not emb_map:
                    pbar.update(len(batch_txts))
                    continue

                # gather embeddings
                embeddings = []
                valid_ids  = []
                for doc_id_i, t in zip(batch_ids, batch_txts):
                    e = emb_map.get(t)
                    if e is not None:
                        embeddings.append(e)
                        valid_ids.append(doc_id_i)

                if embeddings:
                    arr_embeddings = np.vstack(embeddings)
                    self.index.add_with_ids(arr_embeddings, np.array(valid_ids, dtype=np.int64))
                    for doc_id_i, t in zip(batch_ids, batch_txts):
                        if t in emb_map:
                            self.doc_db[str(doc_id_i)] = {"doc": t}
                pbar.update(len(batch_txts))

        if save:
            self.save()

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Dot-product search in float space (IndexFlatIP).
        """
        if self.index.ntotal == 0:
            logger.warning("No docs in index, add documents first.")
            return []

        emb_map = self._generate_float_embeddings([query], input_type="search_query")
        if not emb_map or query not in emb_map:
            logger.error("Query embedding generation failed.")
            return []

        qvec = emb_map[query].reshape(1, -1)
        distances, ids = self.index.search(qvec, k)
        results = []
        for dist, did in zip(distances[0], ids[0]):
            if did == -1:
                continue
            doc_data = self.doc_db.get(str(did), {})
            txt = doc_data.get("doc", "N/A")
            results.append({
                "doc_id": did,
                "score": float(dist),
                "doc": txt
            })
        # By default, Faiss IP index returns highest dot product first. 
        # If you want them descending, verify or re-sort by dist descending.
        # For IP: bigger => more similar
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def remove_document(self, doc_id: int, save: bool = True):
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_db:
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            del self.doc_db[doc_id_str]
        if save:
            self.save()

    def save(self):
        faiss_index_path = os.path.join(self.folder, "index.faiss")
        faiss.write_index(self.index, faiss_index_path)
        logger.info("Float FAISS index saved to disk.")

    def __len__(self):
        return self.index.ntotal