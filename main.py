# main.py
# Copyright© 2025 Constantine Vassilev. All rights reserved
import logging
import os
import numpy as np
import shutil
import statistics
import pandas as pd
from itertools import zip_longest
import matplotlib.pyplot as plt
# ----------------- Local per-document quantization -----------------
from VectorDBInt8 import VectorDBInt8
from VectorDBInt16 import VectorDBInt16
from VectorDBInt4 import VectorDBInt4

# ----------------- Global-limit quantization -----------------
from VectorDBInt8Global import VectorDBInt8Global
from VectorDBInt16Global import VectorDBInt16Global
from VectorDBInt4Global import VectorDBInt4Global

# Example embedding service for float32 inference
from embedding_service import EmbeddingService

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Delete the 'img' folder if it exists
img_folder = 'img'
if os.path.exists(img_folder):
    logger.info(f"Removing existing directory: {img_folder}")
    shutil.rmtree(img_folder, ignore_errors=True)

logger.info(f"Creating new directory: {img_folder}")
os.makedirs(img_folder, exist_ok=True)

# Load DOCS dynamically from the CSV
CSV_FILE = "Generated_AI_Examples.csv"

# Read the file and extract the column into DOCS
try:
    df = pd.read_csv(CSV_FILE)
    DOCS = df["Generated Examples"].tolist()
    DOC_IDS = list(range(len(DOCS)))  # Regenerate DOC_IDS based on new DOCS
    logger.info(f"Loaded {len(DOCS)} documents from {CSV_FILE}.")
except Exception as e:
    logger.error(f"Failed to load {CSV_FILE}: {e}")
    DOCS = []  # Fallback to an empty list if the file is not found
    DOC_IDS = []

# ----------------- Constants & Config -----------------
DB_FOLDER_INT8           = "./db_int8"
DB_FOLDER_INT16          = "./db_int16"
DB_FOLDER_INT4           = "./db_int4"

DB_FOLDER_INT8_GLOBAL    = "./db_int8_global"
DB_FOLDER_INT16_GLOBAL   = "./db_int16_global"
DB_FOLDER_INT4_GLOBAL    = "./db_int4_global"

MODEL_NAME   = "snowflake-arctic-embed2"
EMBEDDING_DIM= 1024
K_RESULTS    = 10
QUERY        = "Artificial intelligence is transforming industries."

def plot_score_comparison(results_float32, results_quantized, labels, file_name):
    doc_ids = [r['doc_id'] for r in results_float32]
    float32_scores = [r['score'] for r in results_float32]
    
    # Ensure results_quantized is a list of lists, even if it's a single list
    if not isinstance(results_quantized[0], list):
        results_quantized = [results_quantized]
    
    quantized_scores_list = []
    for quantized in results_quantized:
        quantized_scores = [next((q['score'] for q in quantized if q['doc_id'] == id), None) for id in doc_ids]
        quantized_scores_list.append(quantized_scores)

    plt.figure(figsize=(12, 6))
    plt.plot(doc_ids, float32_scores, label='Float32', marker='o')
    for scores, label in zip(quantized_scores_list, labels):
        if scores[0] is not None:  # Check if at least one score exists
            plt.plot(doc_ids, scores, label=label, marker='x')
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

# Helper to save results to CSV
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
                logger.warning(f"Could not convert score to float for doc_id {doc_id}")
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
            'Difference': diff if diff != 'N/A' else diff,
            'Percentage_Difference': perc_diff if perc_diff != 'N/A' else perc_diff
        })

    df = pd.DataFrame(data)

    # Check if the CSV file already exists to decide whether to append or write new
    if os.path.exists(csv_file):
        # Read existing data to check for duplicates
        existing_df = pd.read_csv(csv_file)
        # Concatenate new data, ensuring no duplicates based on method and doc_id
        df = pd.concat([existing_df, df]).drop_duplicates(subset=['Method', 'Doc_ID'], keep='last')
        df.to_csv(csv_file, mode='w', index=False)  # Overwrite with updated data
    else:
        df.to_csv(csv_file, mode='w', index=False)

    logger.info(f"Results saved for {method_name} to {csv_file}")

# ----------------- Helper: Show Scores Side by Side -----------------
def show_scores_side_by_side(results_float32: list, results_two_stage: list, label: str):
    """
    Merge doc IDs from both float32 and two-stage results.
    Show them side by side with their scores (or None if a doc doesn't appear in one set),
    including the percentage difference.
    """
    # Build maps for quick lookup
    float_map = {r["doc_id"]: {"score": r["score"], "doc": r["doc"]} for r in results_float32}
    two_map = {r["doc_id"]: {"score": r["score"], "doc": r["doc"]} for r in results_two_stage}

    # Union of all doc IDs
    all_ids = float_map.keys() | two_map.keys()

    # Merge the results
    merged_list = []
    for doc_id in all_ids:
        float_entry = float_map.get(doc_id)
        two_entry = two_map.get(doc_id)
        fs = float_entry["score"] if float_entry else None
        qs = two_entry["score"] if two_entry else None
        doc_text = (float_entry or two_entry)["doc"] if (float_entry or two_entry) else "N/A"
        merged_list.append((doc_id, fs, qs, doc_text))

    # Sort the merged list primarily by float32 score if present, else by the two-stage score
    merged_list.sort(
        key=lambda x: x[1] if x[1] is not None else (x[2] if x[2] is not None else -float('inf')),
        reverse=True
    )

    # Initialize statistics
    percentage_diffs = []

    # Print the query once at the beginning
    logger.info("QUERY:")
    logger.info(QUERY)

    # Print out the results
    logger.info(f"\n=== Detailed Score Comparison: Float32 vs. {label} ===")
    for doc_id, fs, qs, doc_text in merged_list:
        if fs is not None and qs is not None:
            diff = abs(fs - qs)
            perc_diff = (diff / abs(fs)) * 100 if fs != 0 else float('inf')
            percentage_diffs.append(perc_diff)
            logger.info(
                f"Doc ID={doc_id}, float32={fs:.8f}, {label}={qs:.8f}, "
                f"diff={diff:.8f}, diff%={perc_diff:.4f}%, doc='{doc_text[:60]}...'"
            )
        elif fs is None and qs is not None:
            logger.info(
                f"Doc ID={doc_id}, float32=None, {label}={qs:.8f}, "
                f"diff%='N/A', doc='{doc_text[:60]}...'"
            )
        elif fs is not None and qs is None:
            logger.info(
                f"Doc ID={doc_id}, float32={fs:.8f}, {label}=None, "
                f"diff%='N/A', doc='{doc_text[:60]}...'"
            )
        else:
            logger.info(
                f"Doc ID={doc_id}, float32=None, {label}=None, "
                f"diff%='N/A', doc='{doc_text[:60]}...'"
            )

    # Calculate and log summary statistics if there are percentage differences
    if percentage_diffs:
        avg_perc_diff = np.mean(percentage_diffs)
        max_perc_diff = np.max(percentage_diffs)
        min_perc_diff = np.min(percentage_diffs)
        median_perc_diff = np.median(percentage_diffs)
        logger.info(f"\n=== Summary of Percentage Differences for {label} ===")
        logger.info(f"Average Percentage Difference: {avg_perc_diff:.4f}%")
        logger.info(f"Median Percentage Difference: {median_perc_diff:.4f}%")
        logger.info(f"Maximum Percentage Difference: {max_perc_diff:.4f}%")
        logger.info(f"Minimum Percentage Difference: {min_perc_diff:.4f}%")
        
        label = label.replace(' ', '_')
        # Plotting
        plot_score_comparison(results_float32, [results_two_stage], [label], f"{label}_scores_comparison.png")
        plot_percentage_differences({label: percentage_diffs}, f"{label}_percentage_diffs.png")
    else:
        logger.info("\nNo matching documents with both Float32 and Quantized scores to compare.")

def compare_results(results_float32: list, results_quantized: list, label: str = "Quantized"):
    """
    Compare documents from float32 and quantized search results in rank order.
    
    - Logs detailed comparison for each document pair, including absolute and percentage differences.
    - Handles cases where the number of results differs between float32 and quantized lists.
    - Logs warnings for mismatched document IDs or missing documents.
    - Computes and logs summary statistics (average, median, max, min) of percentage differences.
    
    Args:
        results_float32 (list): List of dictionaries containing float32 search results.
        results_quantized (list): List of dictionaries containing quantized search results.
        label (str, optional): Label for the quantized results (e.g., "Int8"). Defaults to "Quantized".
    """
    logger.info(f"\nComparison of Scores (Float32 vs. {label}):")
    percentage_diffs = []
    comparison_data = []

    # Use zip_longest to handle unequal list lengths
    for float32_result, q_result in zip_longest(results_float32, results_quantized, fillvalue=None):
        if float32_result and q_result:
            if float32_result['doc_id'] == q_result['doc_id']:
                score_diff = abs(float32_result['score'] - q_result['score'])
                if float32_result['score'] != 0:
                    perc_diff = (score_diff / abs(float32_result['score'])) * 100
                else:
                    # If both scores are zero, the difference is 0%; else, it's infinite
                    perc_diff = 0.0 if q_result['score'] == 0 else float('inf')
                
                # Append only finite percentage differences
                if np.isfinite(perc_diff):
                    percentage_diffs.append(perc_diff)
                else:
                    logger.warning(
                        f"Percentage difference is infinite for Doc ID={float32_result['doc_id']} "
                        f"due to zero float32 score with non-zero quantized score."
                    )
                
                logger.info(
                    f"Doc ID: {float32_result['doc_id']}, "
                    f"Float32 Score: {float32_result['score']:.8f}, "
                    f"{label} Score: {q_result['score']:.8f}, "
                    f"Difference: {score_diff:.8f}, "
                    f"Difference%: {perc_diff:.4f}%"
                )
            else:
                logger.warning(
                    f"Document IDs do not match: Float32 Doc ID={float32_result['doc_id']} "
                    f"vs Quantized Doc ID={q_result['doc_id']}."
                )
        elif float32_result and not q_result:
            logger.warning(
                f"Quantized result missing for Float32 Doc ID={float32_result['doc_id']}."
            )
        elif q_result and not float32_result:
            logger.warning(
                f"Float32 result missing for Quantized Doc ID={q_result['doc_id']}."
            )
        # If both are None, do nothing

    # Calculate and log summary statistics
    if percentage_diffs:
        perc_diffs_np = np.array(percentage_diffs)
        avg_perc_diff = np.mean(perc_diffs_np)
        median_perc_diff = np.median(perc_diffs_np)
        max_perc_diff = np.max(perc_diffs_np)
        min_perc_diff = np.min(perc_diffs_np)
        logger.info(f"\n=== Summary of Percentage Differences for {label} ===")
        logger.info(f"Average Percentage Difference: {avg_perc_diff:.4f}%")
        logger.info(f"Median Percentage Difference: {median_perc_diff:.4f}%")
        logger.info(f"Maximum Percentage Difference: {max_perc_diff:.4f}%")
        logger.info(f"Minimum Percentage Difference: {min_perc_diff:.4f}%")
        
        label = label.replace(' ', '_')
        # Plotting
        plot_score_comparison(results_float32, results_quantized, [label], f"{label}_scores_comparison.png")
        plot_percentage_differences({label: percentage_diffs}, f"{label}_percentage_diffs.png")
    else:
        logger.info("No matching documents to compare.")

# ----------------- Utility: Cleanup a Folder -----------------
def cleanup_folder(folder_path: str):
    """
    Safely remove the folder_path if it exists, including all subdirectories.
    If folder_path does not exist, log that it's not found and return.
    """
    if not os.path.exists(folder_path):
        logger.info(f"Directory not found: {folder_path}, nothing to remove.")
        return

    logger.info(f"Removing existing directory: {folder_path}")
    shutil.rmtree(folder_path, ignore_errors=True)

# Function to compare selected methods
def compare_selected_methods():
    methods = [
        ("Int4", run_vector_db_int4),
        ("Int4Global", run_vector_db_int4_global),
        ("Int8", run_vector_db_int8),
        ("Int8Global", run_vector_db_int8_global),
        ("Int16", run_vector_db_int16),
        ("Int16Global", run_vector_db_int16_global)
    ]
    percentage_diffs = {}

    # Delete previous results.csv if it exists
    results_csv_path = "results.csv"
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    for method_name, method_func in methods:
        logger.info(f"\nStarting {method_name} Method...")
        method_func()
        logger.info(f"Completed {method_name} Method.")

        # Collecting percentage differences for visualization
        df = pd.read_csv(results_csv_path)
        method_diffs = df[df['Method'] == method_name]['Percentage_Difference']
        percentage_diffs[method_name] = method_diffs[method_diffs != 'N/A'].astype(float).tolist()

    # Visualize percentage differences
    plot_percentage_differences(percentage_diffs, 'percentage_diffs_comparison.png')

    # Visualize score comparison - this would require a bit more work to compare all methods on one plot
    # For simplicity, we'll plot each against Float32 in separate plots
    for method_name in percentage_diffs.keys():
        df_method = df[df['Method'] == method_name]
        float32_results = df_method['Float32_Score'].tolist()
        quantized_results = df_method[f'{method_name}_Score'].tolist()
        plot_score_comparison(
            [{'doc_id': i, 'score': score} for i, score in enumerate(float32_results)],
            [{'doc_id': i, 'score': score} for i, score in enumerate(quantized_results)],
            [method_name],
            f'{method_name}_score_comparison.png'
        )

# ----------------- Functions to Run Each Method -----------------
# ... (keep existing functions like run_vector_db_int8, run_vector_db_int16, etc., 
#     but add save_to_csv calls in each function)
def run_vector_db_int8():
    logger.info("=== Single-Stage Int8 (Local) ===")
    cleanup_folder(DB_FOLDER_INT8)
    vector_db = VectorDBInt8(folder=DB_FOLDER_INT8, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)

    logger.info("Adding documents to Int8 DB...")
    vector_db.add_documents(doc_ids=DOC_IDS, docs=DOCS)

    # Two searches: float32 vs. int8
    results_float32 = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=True)
    results_int8    = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=False)

    logger.info("Search Results (Float32 vs. Int8):")
    logger.info("QUERY:")
    logger.info(QUERY)
    compare_results(results_float32, results_int8, label="Int8")
    show_scores_side_by_side(results_float32, results_int8, label="Int8 Side By Side")
    save_to_csv(results_float32, results_int8, "Int8")

def run_vector_db_int16():
    logger.info("=== Single-Stage Int16 (Local) ===")
    cleanup_folder(DB_FOLDER_INT16)
    vector_db = VectorDBInt16(folder=DB_FOLDER_INT16, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)

    logger.info("Adding documents to Int16 DB...")
    vector_db.add_documents(doc_ids=DOC_IDS, docs=DOCS)

    # Two searches: float32 vs. int16
    results_float32 = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=True)
    results_int16   = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=False)

    logger.info("Search Results (Float32 vs. Int16):")
    logger.info("QUERY:")
    logger.info(QUERY)
    compare_results(results_float32, results_int16, label="Int16")
    show_scores_side_by_side(results_float32, results_int16, label="Int16 Side By Side")
    save_to_csv(results_float32, results_int16, "Int16")

def run_vector_db_int4():
    logger.info("=== Single-Stage Int4 (Local) ===")
    cleanup_folder(DB_FOLDER_INT4)
    vector_db = VectorDBInt4(folder=DB_FOLDER_INT4, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM)

    logger.info("Adding documents to Int4 DB...")
    vector_db.add_documents(doc_ids=DOC_IDS, docs=DOCS)

    # Two searches: float32 vs. int4
    results_float32 = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=True)
    results_int4    = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=False)

    logger.info("Search Results (Float32 vs. Int4):")
    logger.info("QUERY:")
    logger.info(QUERY)
    compare_results(results_float32, results_int4, label="Int4")
    show_scores_side_by_side(results_float32, results_int4, label="Int4 Side By Side")
    save_to_csv(results_float32, results_int4, "Int4")

def run_vector_db_int8_global():
    logger.info("=== Single-Stage Int8 (Global) ===")
    cleanup_folder(DB_FOLDER_INT8_GLOBAL)
    vector_db = VectorDBInt8Global(folder=DB_FOLDER_INT8_GLOBAL, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM, global_limit=0.3)

    logger.info("Adding documents to Int8Global DB...")
    logger.info("QUERY:")
    logger.info(QUERY)
    vector_db.add_documents(doc_ids=DOC_IDS, docs=DOCS)

    # Two searches: float32 vs. int8_global
    results_float32 = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=True)
    results_int8    = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=False)

    logger.info("Search Results (Float32 vs. Int8Global):")
    logger.info("QUERY:")
    logger.info(QUERY)
    compare_results(results_float32, results_int8, label="Int8Global")
    show_scores_side_by_side(results_float32, results_int8, label="Int8Global Side By Side")
    save_to_csv(results_float32, results_int8, "Int8Global")

def run_vector_db_int16_global():
    logger.info("=== Single-Stage Int16 (Global) ===")
    cleanup_folder(DB_FOLDER_INT16_GLOBAL)
    vector_db = VectorDBInt16Global(folder=DB_FOLDER_INT16_GLOBAL, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM, global_limit=1.0)

    logger.info("Adding documents to Int16Global DB...")
    vector_db.add_documents(doc_ids=DOC_IDS, docs=DOCS)

    # Two searches: float32 vs. int16_global
    results_float32 = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=True)
    results_int16   = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=False)

    logger.info("Search Results (Float32 vs. Int16Global):")
    logger.info("QUERY:")
    logger.info(QUERY)
    compare_results(results_float32, results_int16, label="Int16Global")
    show_scores_side_by_side(results_float32, results_int16, label="Int16Global Side By Side")
    save_to_csv(results_float32, results_int16, "Int16Global")

def run_vector_db_int4_global():
    logger.info("=== Single-Stage Int4 (Global) ===")
    cleanup_folder(DB_FOLDER_INT4_GLOBAL)
    vector_db = VectorDBInt4Global(folder=DB_FOLDER_INT4_GLOBAL, model=MODEL_NAME, embedding_dim=EMBEDDING_DIM, global_limit=0.18)

    logger.info("Adding documents to Int4Global DB...")
    vector_db.add_documents(doc_ids=DOC_IDS, docs=DOCS)

    # Two searches: float32 vs. int4_global
    results_float32 = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=True)
    results_int4    = vector_db.search(query=QUERY, k=K_RESULTS, compare_float32=False)

    logger.info("Search Results (Float32 vs. Int4Global):")
    logger.info("QUERY:")
    logger.info(QUERY)
    compare_results(results_float32, results_int4, label="Int4Global")
    show_scores_side_by_side(results_float32, results_int4, label="Int4Global Side By Side")
    save_to_csv(results_float32, results_int4, "Int4Global")
    
    # ----------------- Main -----------------
if __name__ == "__main__":
    """
    This main script includes:
      1) Comparison of selected quantization methods (Int4, Int4Global, Int8, Int8Global, Int16, Int16Global).
      2) Saving results to CSV and generating visualizations.

    Running selected methods:
    """
    compare_selected_methods()