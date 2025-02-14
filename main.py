# main.py
# Copyright© 2025 Constantine Vassilev. All rights reserved

"""
Detailed Description:
---------------------

This script serves as the main entry point for benchmarking and comparing various vector database implementations
that leverage Cohere’s embedding API along with different quantization techniques and indexing strategies.
The primary goals of the script are to:

1. Load documents from a CSV file.
2. Build several types of vector databases:
   - Local per-document quantization (Int8, Int16, Int4)
   - Global-limit quantization (Int8Global, Int16Global, Int4Global)
   - Cohere-specific quantization using int8 embeddings and signed binary representations.
   - Full float-based embeddings (if available) using CohereVectorDBFloat.
   - An enhanced multi-phase search database (CohereEnhancedVectorDB) which combines int8, ubinary, and float embeddings.
3. Execute search queries using these databases and compare their performance in terms of:
   - Build time (time taken to index the documents)
   - Search time (time taken to retrieve search results)
   - Storage size of the built indexes
4. Optionally perform a rerank step using Cohere’s rerank API to refine the search results.
5. Compare the top results from the float-based database against the quantized databases and generate summary statistics.
6. Generate plots and CSV outputs to visualize the score comparisons and percentage differences.

Workflow Overview:
------------------
- **Initialization and Setup:**  
  - The script begins by cleaning up an output directory (for generated images).
  - Documents are loaded from a CSV file named "Generated_AI_Examples.csv".
  - Global constants such as database folder paths, the embedding model name, embedding dimensions, query string, and the number of results are defined.

- **Plotting and Utility Functions:**  
  - Functions are provided to plot score comparisons and percentage differences, save CSV reports, normalize scores, and calculate folder sizes.

- **Building and Querying Vector Databases:**  
  - Each database variant (Float, Int8, Enhanced, etc.) has its dedicated function to:
      1. Build the database by indexing documents and generating embeddings.
      2. Execute search queries and measure the performance.
      3. Log detailed metrics including build time, search time, and database size.
  - For int8-based approaches, scores are normalized against the float-based results to allow fair comparison.

- **Reranking:**  
  - An additional function demonstrates how to use Cohere’s rerank API to reorder a candidate list of results.

- **Comparison and Summary:**  
  - The script compares the top search results from the float-based and int8-based databases.
  - A final summary of all performance metrics is logged, including percentage improvements in build/search times and index sizes.

Usage:
------
Ensure the following environment variables are set before running the script:
    - COHERE_EMBED_ENDPOINT: URL for the Cohere embedding API.
    - COHERE_EMBED_KEY: API key for the Cohere service.
    - (Optional) COHERE_RERANK_ENDPOINT and COHERE_RERANK_KEY for reranking.

To run the script:
    python main.py

Results (plots and CSV) are stored in the output folder ("img") and a CSV file ("results.csv").

"""

import logging
import os
import json
import time
import numpy as np
import faiss
import shutil
import statistics
import pandas as pd
from itertools import zip_longest
import matplotlib.pyplot as plt
import requests
from rocksdict import Rdict
from typing import List, Dict

logger = logging.getLogger(__name__)

# ----------------- Local per-document quantization -----------------
#from VectorDBInt8 import VectorDBInt8
#from VectorDBInt16 import VectorDBInt16
#from VectorDBInt4 import VectorDBInt4

# ----------------- Global-limit quantization -----------------
#from VectorDBInt8Global import VectorDBInt8Global
#from VectorDBInt16Global import VectorDBInt16Global
#from VectorDBInt4Global import VectorDBInt4Global

# ----------------- Cohere Specific Quantization -----------------
from CohereVectorDBInt8 import CohereVectorDBInt8
from CohereVectorDBBinary import CohereVectorDBBinary

# Example embedding service for float32 inference
from embedding_service import EmbeddingService

########################################################################
# We'll import "CohereVectorDBFloat" from "CohereVectorDBFloat.py"
# to do full float-based embeddings from Cohere.
########################################################################
try:
    from CohereVectorDBFloat import CohereVectorDBFloat
    HAVE_COHERE_FLOAT = True
except ImportError:
    HAVE_COHERE_FLOAT = False

# ----------------- New: Enhanced Cohere DB -----------------
from CohereEnhancedVectorDB import CohereEnhancedVectorDB

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

########################################################################
# 1) Clean up "img" folder
########################################################################
img_folder = 'img'
if os.path.exists(img_folder):
    logger.info(f"Removing existing directory: {img_folder}")
    shutil.rmtree(img_folder, ignore_errors=True)
logger.info(f"Creating new directory: {img_folder}")
os.makedirs(img_folder, exist_ok=True)

########################################################################
# 2) Load Documents from CSV
########################################################################
CSV_FILE = "Generated_AI_Examples.csv"
try:
    df = pd.read_csv(CSV_FILE)
    DOCS = df["Generated Examples"].tolist()
    DOC_IDS = list(range(len(DOCS)))
    logger.info(f"Loaded {len(DOCS)} documents from {CSV_FILE}.")
except Exception as e:
    logger.error(f"Failed to load {CSV_FILE}: {e}")
    DOCS = []
    DOC_IDS = []

########################################################################
# 3) Global Constants
########################################################################
DB_FOLDER_INT8            = "./db_int8"
DB_FOLDER_INT16           = "./db_int16"
DB_FOLDER_INT4            = "./db_int4"

DB_FOLDER_INT8_GLOBAL     = "./db_int8_global"
DB_FOLDER_INT16_GLOBAL    = "./db_int16_global"
DB_FOLDER_INT4_GLOBAL     = "./db_int4_global"

DB_FOLDER_COHERE_INT8     = "./db_cohere_int8"
DB_FOLDER_COHERE_BINARY   = "./db_cohere_binary"
DB_FOLDER_COHERE_FLOAT    = "./db_cohere_float"  # For float-based approach

DB_FOLDER_COHERE_ENHANCED = "./db_cohere_enhanced"  # For the enhanced version

MODEL_NAME    = "embed-english-v3.0"
EMBEDDING_DIM = 1024
K_RESULTS     = 50
QUERY         = "Artificial intelligence is transforming industries."

########################################################################
# 4) Plotting & CSV Utilities
########################################################################
def plot_score_comparison(results_float32, results_quantized, labels, file_name):
    doc_ids = [r['doc_id'] for r in results_float32]
    float32_scores = [r['score'] for r in results_float32]
    if not isinstance(results_quantized[0], list):
        results_quantized = [results_quantized]

    quantized_scores_list = []
    for quantized in results_quantized:
        quantized_scores = [next((q['score'] for q in quantized if q['doc_id'] == did), None) for did in doc_ids]
        quantized_scores_list.append(quantized_scores)

    plt.figure(figsize=(12, 6))
    plt.plot(doc_ids, float32_scores, label='Float32', marker='o')
    for scores, lab in zip(quantized_scores_list, labels):
        if scores and scores[0] is not None:
            plt.plot(doc_ids, scores, label=lab, marker='x')

    plt.xlabel('Document ID')
    plt.ylabel('Score')
    plt.title(f'Score Comparison: Float32 vs {", ".join(labels)}')
    plt.legend()
    plt.grid(True)
    os.makedirs('img', exist_ok=True)
    plt.savefig(f'img/{file_name}')
    plt.close()

def plot_percentage_differences(percentage_diffs_dict, file_name):
    plt.figure(figsize=(12, 6))
    for method, diffs in percentage_diffs_dict.items():
        plt.hist(diffs, bins=20, alpha=0.5, label=method, edgecolor='black')
    plt.xlabel('Percentage Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Percentage Differences')
    plt.legend()
    plt.grid(True)
    os.makedirs('img', exist_ok=True)
    plt.savefig(f'img/{file_name}')
    plt.close()

def save_to_csv(results_float32, results_quantized, method_name, csv_file="results.csv"):
    if not results_float32 or not results_quantized:
        logger.warning(f"No results to save for method: {method_name}")
        return

    data = []
    for float32_result, quantized_result in zip_longest(results_float32, results_quantized, fillvalue={}):
        doc_id = float32_result.get('doc_id', quantized_result.get('doc_id', 'N/A'))
        float32_score = float32_result.get('score', 'N/A')
        quantized_score = quantized_result.get('score', 'N/A')

        if float32_score != 'N/A' and quantized_score != 'N/A':
            try:
                float32_score = float(float32_score)
                quantized_score = float(quantized_score)
                diff = abs(float32_score - quantized_score)
                perc_diff = (diff / abs(float32_score)) * 100 if float32_score != 0 else float('inf')
            except ValueError:
                diff = 'N/A'
                perc_diff = 'N/A'
        else:
            diff = 'N/A'
            perc_diff = 'N/A'

        data.append({
            'Method': method_name,
            'Doc_ID': doc_id,
            'Float32_Score': float32_score,
            f'{method_name}_Score': quantized_score,
            'Difference': diff,
            'Percentage_Difference': perc_diff
        })

    df = pd.DataFrame(data)
    csv_exists = os.path.exists(csv_file)
    if csv_exists:
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df]).drop_duplicates(subset=['Method', 'Doc_ID'], keep='last')
        df.to_csv(csv_file, mode='w', index=False)
    else:
        df.to_csv(csv_file, mode='w', index=False)

    logger.info(f"Results saved for {method_name} to {csv_file}")

########################################################################
# 4a) Helper: Get directory size in bytes
########################################################################
def get_directory_size(directory: str) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

########################################################################
# 4b) Helper: Normalize results via min-max mapping
########################################################################
def normalize_results(results, target_min, target_max):
    raw_scores = [r['score'] for r in results]
    raw_min = min(raw_scores)
    raw_max = max(raw_scores)
    if raw_max == raw_min:
        return results
    normalized = []
    for r in results:
        norm_score = (r['score'] - raw_min) / (raw_max - raw_min) * (target_max - target_min) + target_min
        new_r = r.copy()
        new_r['score'] = norm_score
        normalized.append(new_r)
    return normalized

########################################################################
# 5) Misc Helpers: compare + log
########################################################################
def cleanup_folder(folder_path: str):
    if os.path.exists(folder_path):
        logger.info(f"Removing existing directory: {folder_path}")
        shutil.rmtree(folder_path, ignore_errors=True)
    else:
        logger.info(f"Directory not found: {folder_path}, nothing to remove.")

########################################################################
# 6) Build & Run: Cohere Float
########################################################################
def run_cohere_vector_db_float():
    logger.info("=== Single-Stage Cohere Float (Benchmark) ===")
    if not HAVE_COHERE_FLOAT:
        logger.warning("CohereVectorDBFloat not available. Skipping float method.")
        return None
    cleanup_folder(DB_FOLDER_COHERE_FLOAT)
    from CohereVectorDBFloat import CohereVectorDBFloat
    float_db = CohereVectorDBFloat(folder=DB_FOLDER_COHERE_FLOAT, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    
    t0 = time.time()
    logger.info("Adding documents to Cohere Float DB...")
    float_db.add_documents(DOC_IDS, DOCS)
    t1 = time.time()
    build_time = t1 - t0
    float_db_size = get_directory_size(DB_FOLDER_COHERE_FLOAT)
    logger.info("Time to build Cohere Float DB: %.2f seconds", build_time)
    logger.info("Cohere Float DB size: %d bytes", float_db_size)
    
    t2 = time.time()
    logger.info("Performing float-based search for the query...")
    results_float = float_db.search(query=QUERY, k=K_RESULTS)
    t3 = time.time()
    search_time = t3 - t2
    logger.info("Time to retrieve Cohere Float results: %.2f seconds", search_time)
    
    logger.info("Search Results (CohereFloat):")
    logger.info(f"QUERY: {QUERY}")
    for r in results_float:
        logger.info(f" DocID={r['doc_id']}, Score={r['score']:.6f}, Doc='{r['doc']}'")
    
    save_to_csv(results_float, results_float, "CohereFloat")
    return {"results": results_float, "build_time": build_time, "search_time": search_time, "db_size": float_db_size}

########################################################################
# 7) Build & Run: Cohere Int8 with Min-Max Normalization
########################################################################
def run_cohere_vector_db_int8():
    logger.info("=== Single-Stage Cohere Int8 (Local) ===")
    cleanup_folder(DB_FOLDER_COHERE_INT8)
    from CohereVectorDBInt8 import CohereVectorDBInt8
    int8_db = CohereVectorDBInt8(folder=DB_FOLDER_COHERE_INT8, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    
    t0 = time.time()
    logger.info("Adding documents to Cohere Int8 DB...")
    int8_db.add_documents(DOC_IDS, DOCS)
    t1 = time.time()
    build_time = t1 - t0
    int8_db_size = get_directory_size(DB_FOLDER_COHERE_INT8)
    logger.info("Time to build Cohere Int8 DB: %.2f seconds", build_time)
    logger.info("Cohere Int8 DB size: %d bytes", int8_db_size)
    
    t2 = time.time()
    results_cohere_int8 = int8_db.search(query=QUERY, k=K_RESULTS)
    t3 = time.time()
    search_time = t3 - t2
    logger.info("Time to retrieve raw Cohere Int8 results: %.2f seconds", search_time)
    
    logger.info("Raw Search Results (Cohere Int8):")
    logger.info(f"QUERY: {QUERY}")
    for r in results_cohere_int8:
        logger.info(f" DocID={r['doc_id']}, Raw Score={r['score']}, Doc='{r['doc']}'")
    
    from CohereVectorDBFloat import CohereVectorDBFloat
    float_db = CohereVectorDBFloat(folder=DB_FOLDER_COHERE_FLOAT, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    if len(float_db) == 0:
        logger.info("Cohere Float DB empty in normalization step. Adding docs now.")
        float_db.add_documents(DOC_IDS, DOCS)
    results_float = float_db.search(query=QUERY, k=K_RESULTS)
    target_scores = [r['score'] for r in results_float]
    target_min, target_max = min(target_scores), max(target_scores)
    logger.info("Target range from CohereFloat: min=%.6f, max=%.6f", target_min, target_max)
    
    normalized_int8_results = normalize_results(results_cohere_int8, target_min, target_max)
    normalized_int8_results = sorted(normalized_int8_results, key=lambda r: r['score'], reverse=True)
    
    logger.info("Normalized Search Results (Cohere Int8):")
    for r in normalized_int8_results:
        logger.info(f" DocID={r['doc_id']}, Normalized Score={r['score']:.6f}, Doc='{r['doc']}'")
    
    save_to_csv(normalized_int8_results, normalized_int8_results, "CohereInt8")
    return {"results": normalized_int8_results, "build_time": build_time, "search_time": search_time, "db_size": int8_db_size}

########################################################################
# 7a) New Method: Search with Rerank using Cohere's Rerank API
########################################################################
def run_cohere_rerank():
    from CohereVectorDBInt8 import CohereVectorDBInt8
    int8_db = CohereVectorDBInt8(folder=DB_FOLDER_COHERE_INT8, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    if len(int8_db) == 0:
        logger.info("Cohere Int8 DB is empty. Adding documents now.")
        int8_db.add_documents(DOC_IDS, DOCS)
    
    t0 = time.time()
    logger.info("Calling search_rerank_cohere()...")
    rerank_results = int8_db.search_rerank_cohere(
        query=QUERY,
        k=50,
        binary_oversample=10,
        rerank_model="rerank-english-v3.0"
    )
    t1 = time.time()
    rerank_time = t1 - t0
    logger.info("Time to retrieve reranked results: %.2f seconds", rerank_time)
    
    logger.info("Reranked Results:")
    for r in rerank_results:
        logger.info(f" DocID={r['doc_id']}, Score={r['score']:.4f}, Doc='{r['doc']}'")
    return {"results": rerank_results, "rerank_time": rerank_time}

########################################################################
# 8) Build & Run: Enhanced Cohere DB (Multi-Phase Search)
########################################################################
def run_cohere_enhanced():
    logger.info("=== Enhanced Cohere DB (Multi-Phase: int8/ubinary/float) ===")
    cleanup_folder(DB_FOLDER_COHERE_ENHANCED)
    enhanced_db = CohereEnhancedVectorDB(folder=DB_FOLDER_COHERE_ENHANCED, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    
    t0 = time.time()
    logger.info("Adding documents to Cohere Enhanced DB...")
    enhanced_db.add_documents(DOC_IDS, DOCS)
    t1 = time.time()
    build_time = t1 - t0
    enhanced_db_size = get_directory_size(DB_FOLDER_COHERE_ENHANCED)
    logger.info("Time to build Cohere Enhanced DB: %.2f seconds", build_time)
    logger.info("Cohere Enhanced DB size: %d bytes", enhanced_db_size)
    
    t2 = time.time()
    logger.info("Performing enhanced search for the query...")
    enhanced_results = enhanced_db.search(query=QUERY, k=K_RESULTS)
    t3 = time.time()
    search_time = t3 - t2
    logger.info("Time to retrieve enhanced results: %.2f seconds", search_time)
    
    logger.info("Enhanced Search Results:")
    logger.info(f"QUERY: {QUERY}")
    for r in enhanced_results:
        logger.info(f" DocID={r['doc_id']}, Score={r.get('score_cosine', r.get('score')):.6f}, Doc='{r['doc']}'")
    
    save_to_csv(enhanced_results, enhanced_results, "CohereEnhanced")
    return {"results": enhanced_results, "build_time": build_time, "search_time": search_time, "db_size": enhanced_db_size}

########################################################################
# 9) Compare Top-10 Float vs. Int8 (Doc by Doc)
########################################################################
def get_top_cohere_float() -> list:
    if not HAVE_COHERE_FLOAT:
        logger.warning("CohereVectorDBFloat not available, returning empty float results.")
        return []
    from CohereVectorDBFloat import CohereVectorDBFloat
    db = CohereVectorDBFloat(folder=DB_FOLDER_COHERE_FLOAT, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    if len(db) == 0:
        logger.info("Cohere Float DB empty. Adding docs now.")
        db.add_documents(DOC_IDS, DOCS)
    results = db.search(query=QUERY, k=50)
    results = sorted(results, key=lambda r: r['score'], reverse=True)
    return results

def get_top_cohere_int8() -> list:
    from CohereVectorDBInt8 import CohereVectorDBInt8
    db = CohereVectorDBInt8(folder=DB_FOLDER_COHERE_INT8, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    if len(db) == 0:
        logger.info("Cohere Int8 DB empty. Adding docs now.")
        db.add_documents(DOC_IDS, DOCS)
    raw_int8_results = db.search(query=QUERY, k=50)
    
    from CohereVectorDBFloat import CohereVectorDBFloat
    float_db = CohereVectorDBFloat(folder=DB_FOLDER_COHERE_FLOAT, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)
    if len(float_db) == 0:
        logger.info("Cohere Float DB empty in normalization step. Adding docs now.")
        float_db.add_documents(DOC_IDS, DOCS)
    float_results = float_db.search(query=QUERY, k=50)
    target_scores = [r['score'] for r in float_results]
    target_min, target_max = min(target_scores), max(target_scores)
    
    normalized_int8_results = normalize_results(raw_int8_results, target_min, target_max)
    normalized_int8_results = sorted(normalized_int8_results, key=lambda r: r['score'], reverse=True)
    return normalized_int8_results

def compare_top_float_and_int8():
    logger.info("\n=== Compare Top-10 Float vs. Int8 ===")
    top_float = get_top_cohere_float()
    logger.info(" --- Top-10 Float ---")
    for f in top_float:
        logger.info(f"  DocID={f['doc_id']}, Score={f['score']:.4f}, Doc='{f['doc']}'")

    top_int8 = get_top_cohere_int8()
    logger.info(" --- Top-10 Int8 ---")
    for i in top_int8:
        logger.info(f"  DocID={i['doc_id']}, Score={i['score']:.4f}, Doc='{i['doc']}'")

    logger.info("\n=== Float vs. Int8 (Top-10 Overlap) ===")
    float_map = {item["doc_id"]: item for item in top_float}
    int8_map  = {item["doc_id"]: item for item in top_int8}
    all_ids   = set(float_map.keys()) | set(int8_map.keys())

    diffs = []
    for did in all_ids:
        f_item = float_map.get(did)
        i_item = int8_map.get(did)
        if f_item and i_item:
            f_score = f_item["score"]
            i_score = i_item["score"]
            diff = abs(f_score - i_score)
            diffs.append(diff)
            logger.info(f"DocID={did}, Float={f_score:.4f}, Int8={i_score:.4f}, Diff={diff:.4f}")
        elif f_item:
            logger.info(f"DocID={did} => ONLY in Float top10, Score={f_item['score']:.4f}")
        elif i_item:
            logger.info(f"DocID={did} => ONLY in Int8 top10, Score={i_item['score']:.4f}")

    if diffs:
        arr = np.array(diffs)
        logger.info(f"\nDifferences Stats: avg={arr.mean():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

########################################################################
# 10) Main: Compare selected methods, then unify top-10, call rerank, and summarize metrics
########################################################################
if __name__ == "__main__":
    def format_time(seconds):
        mins = int(seconds // 60)
        secs = seconds % 60
        if mins > 0:
            return f"{mins} min {secs:.2f} sec"
        else:
            return f"{secs:.2f} sec"

    def format_size(bytes_):
        mb = bytes_ / (1024 * 1024)
        return f"{mb:.2f} MB"

    # Run the selected methods and collect performance metrics.
    float_metrics    = run_cohere_vector_db_float()      # Cohere Float
    int8_metrics     = run_cohere_vector_db_int8()         # Cohere Int8 with normalization
    enhanced_metrics = run_cohere_enhanced()               # New Enhanced Cohere DB
    rerank_metrics   = run_cohere_rerank()                 # Cohere rerank using API

    compare_top_float_and_int8()

    # Summary:
    logger.info("\n=== Summary of Performance Metrics ===")
    if float_metrics and int8_metrics and enhanced_metrics and rerank_metrics:
        float_build = float_metrics["build_time"]
        int8_build = int8_metrics["build_time"]
        enhanced_build = enhanced_metrics["build_time"]
        build_gain = ((float_build - int8_build) / float_build) * 100

        float_search = float_metrics["search_time"]
        int8_search = int8_metrics["search_time"]
        enhanced_search = enhanced_metrics["search_time"]
        search_gain = ((float_search - int8_search) / float_search) * 100

        float_size = float_metrics["db_size"]
        int8_size = int8_metrics["db_size"]
        enhanced_size = enhanced_metrics["db_size"]
        size_gain = ((float_size - int8_size) / float_size) * 100

        rerank_time = rerank_metrics["rerank_time"]

        logger.info("Float DB Build Time:    %s", format_time(float_build))
        logger.info("Int8 DB Build Time:     %s (%.2f%% faster)", format_time(int8_build), build_gain)
        logger.info("Enhanced DB Build Time: %s", format_time(enhanced_build))
        logger.info("Float DB Search Time:   %s", format_time(float_search))
        logger.info("Int8 DB Search Time:    %s (%.2f%% faster)", format_time(int8_search), search_gain)
        logger.info("Enhanced DB Search Time:%s", format_time(enhanced_search))
        logger.info("Float DB Size:          %s", format_size(float_size))
        logger.info("Int8 DB Size:           %s (%.2f%% smaller)", format_size(int8_size), size_gain)
        logger.info("Enhanced DB Size:       %s", format_size(enhanced_size))
        logger.info("Rerank Time:            %s", format_time(rerank_time))
    else:
        logger.info("One or more performance metrics are missing.")