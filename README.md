# Cohere Vector DB Benchmarking


**Summary**  
This project evaluates different approaches to vector search, focusing on how Cohere’s embedding API can integrate with FAISS, RocksDB (via RocksDict), and various quantization techniques (e.g., int8, binary) to balance accuracy, performance, and storage needs. Key highlights include:

1. **Embedding Generation with Cohere:**  
   - Documents and queries are embedded using Cohere’s API.  
   - Float, int8, and binary (ubinary) embeddings are supported.

2. **FAISS Integration:**  
   - **Float** embeddings use a traditional FAISS index (e.g., `IndexFlatIP` for dot-product similarity).  
   - **Binary** embeddings use a `faiss.IndexBinaryIDMap2` for Hamming-distance-based approximate nearest-neighbor search.

3. **RocksDB Storage (via RocksDict):**  
   - Document texts and int8 embeddings are stored persistently, minimizing memory usage and eliminating re-indexing on repeated runs.

4. **Enhanced Multi-Phase Search:**  
   - **Phase I:** Quick retrieval with a FAISS binary index (ubinary embeddings).  
   - **Phase II:** Dot-product rescoring with the query’s float embedding.  
   - **Phase III:** Final cosine similarity ranking with stored int8 embeddings.

5. **Benchmark Results:**  
   - **Float Approach (CohereFloat):** Larger index, longer build time, but full precision.  
   - **Enhanced Approach (CohereEnhancedVectorDB):** Smaller index (around 66% reduction), ~45% faster build time, and nearly identical top search results.

6. **Memory & Cost Savings:**  
   - Switching to int8 or binary can yield ~75% (or greater) storage reduction with minimal drop in search quality.  
   - Savings translate to significantly lower hosting costs and improved scalability.

---


---

## Table of Contents
1. [Overview](#overview)
2. [How Cohere is Used](#how-cohere-is-used)
3. [How FAISS and Cohere Binary are Used](#how-faiss-and-cohere-binary-are-used)
4. [How RocksDB is Used](#how-rocksdb-is-used)
5. [Int8 and Binary Embeddings: Memory & Cost Savings](#int8-and-binary-embeddings-memory--cost-savings)
6. [CohereEnhancedVectorDB](#cohereenhancedvectordb)
7. [Benchmark Results and Analysis](#benchmark-results-and-analysis)
   - [Float-Based Search (CohereFloat)](#float-based-search-coherefloat)
   - [Enhanced Multi-Phase Search (CohereEnhancedVectorDB)](#enhanced-multi-phase-search-cohereenhancedvectordb)
   - [Comparison and Analysis](#comparison-and-analysis)
8. [Project Structure](#project-structure)
9. [Requirements](#requirements)
10. [Environment Variables](#environment-variables)
11. [Usage](#usage)
12. [Results](#results)
13. [License](#license)


---

## Overview

The project implements several vector database approaches:

- **Local Per-Document Quantization:**  
  Uses modules such as `VectorDBInt8`, `VectorDBInt16`, and `VectorDBInt4` to quantize individual documents.

- **Global-Limit Quantization:**  
  Implements global quantization variants via `VectorDBInt8Global`, `VectorDBInt16Global`, and `VectorDBInt4Global`.

- **Cohere-Specific Quantization:**  
  - **Int8 & Binary Approaches:**  
    Uses `CohereVectorDBInt8` and `CohereVectorDBBinary` to generate int8 embeddings and pack them into compact binary representations.
  - **Float-Based Approach:**  
    Uses `CohereVectorDBFloat` (if available) to generate full-precision float embeddings.

- **Enhanced Multi-Phase Search (CohereEnhancedVectorDB):**  
  Combines multiple embedding types (int8, ubinary, and float) using a three-phase search process:
  1. **Phase I – Fast Retrieval:** Retrieves candidate documents using a FAISS binary search.
  2. **Phase II – Dot-Product Rescoring:** Computes a dot-product between the query’s float representation and unpacked binary embeddings.
  3. **Phase III – Cosine Similarity Rescoring:** Finalizes ranking using cosine similarity with stored int8 embeddings.

- **Reranking Using Cohere's Rerank API:**  
  An optional step to further reorder search results using an external rerank API.

---

## How Cohere is Used

This project integrates Cohere’s embedding API to generate various text embeddings required for indexing and search:

- **HTTP-Based Embedding Generation:**  
  Each time a document is indexed or a search query is processed, an HTTP POST request is sent to Cohere’s embedding API. The endpoint is specified by the `COHERE_EMBED_ENDPOINT` environment variable and is authenticated using the API key from `COHERE_EMBED_KEY`.

- **Multiple Embedding Types:**  
  The API requests specify a list of desired embedding types (e.g., `int8`, `ubinary`, and `float`):
  - **int8 Embeddings:**  
    Provide compact representations that are later used for final cosine similarity rescoring.
  - **ubinary Embeddings:**  
    Derived from int8 embeddings via thresholding and bit-packing (using NumPy’s `packbits`), these are used for fast FAISS binary indexing.
  - **float Embeddings:**  
    Offer full-precision representations that are used during search to compute refined similarity scores.

- **Response Processing:**  
  The embeddings returned by Cohere are parsed from the JSON response. The int8 embeddings are stored with the document text in a RocksDict database, and the ubinary embeddings are used to update the FAISS binary index.

- **Search Workflow Integration:**  
  For a search query, both float and ubinary embeddings are generated. The ubinary embedding enables rapid candidate retrieval via FAISS (Phase I), while the float embedding is used in later rescoring phases (Phases II and III).

---

## How FAISS and Cohere Binary are Used

- **FAISS Indexing:**  
  FAISS (Facebook AI Similarity Search) is used to build a fast binary index that supports approximate nearest-neighbor search. The ubinary embeddings—derived from the int8 embeddings through thresholding and bit-packing—are indexed using a FAISS binary index (wrapped by `faiss.IndexBinaryIDMap2`). This index leverages Hamming distance to rapidly identify candidate documents.

- **Efficient Search:**  
  During the search process, the query’s ubinary embedding is used for a binary search on the FAISS index, rapidly retrieving an oversampled candidate set. This candidate set is then refined using dot-product and cosine similarity calculations.

- **Cohere Binary (ubinary) Conversion:**  
  The process involves converting the int8 embeddings into a binary format (ubinary) by comparing each element to its mean value and then using NumPy’s `packbits` function to create a compact binary representation. This conversion is key to achieving fast indexing and search without a significant loss in accuracy.

---

## How RocksDB is Used

The project uses **RocksDict** (built on RocksDB) as a lightweight, persistent key–value store to manage document storage:

- **Persistent Storage:**  
  Each document’s text and its corresponding int8 embedding are stored in RocksDict. This ensures that data persists between runs, reducing the need to re-index documents.

- **Efficient Retrieval:**  
  When performing searches, document data is quickly retrieved from RocksDict, enabling rapid access to the document texts and their int8 embeddings for rescoring.

- **Integration with the Search Process:**  
  RocksDict serves as the backend for storing document information in modules such as `CohereEnhancedVectorDB`, helping maintain a low memory footprint and supporting scalability.

---

## Int8 and Binary Embeddings: Memory & Cost Savings

According to data shared by [Cohere’s blog](https://cohere.com/blog/int8-binary-embeddings) on indexing 250M Wikipedia embeddings:

| **Model**               | **Search Quality (nDCG@10)**<br/>_(Semantic Search on 18 languages, higher = better)_ | **Memory needed for 250M embeddings** | **Estimated Price on AWS**<br/>_(using x2gd @ \$3.8 per GB/mo)_ |
|-------------------------|:-----------------------------------------------------------------------------------:|:-------------------------------------:|:--------------------------------------------------------------:|
| **Cohere Embed v3**     |                                                                                     |                                       |                                                                  |
| Embed v3                | **66.3**                                                                            | **954 GB**                            | **\$43,488 / yr**                                               |
| Embed v3 - int8         | **66.1**                                                                            | **238 GB**                            | **\$10,872 / yr**                                               |
| Embed v3 - binary       | **62.8**                                                                            | **30 GB**                             | **\$1,359 / yr**                                                |
| **Other Models**        |                                                                                     |                                       |                                                                  |
| OpenAI ada-002          | 31.4                                                                                | 1431 GB                               | \$65,231 / yr                                                   |
| OpenAI 3-small          | 44.9                                                                                | 1431 GB                               | \$65,231 / yr                                                   |
| OpenAI 3-large          | 54.9                                                                                | 2861 GB                               | \$130,463 / yr                                                  |

**Key Takeaways:**
- **Full-Precision vs. Int8:**  
  Moving from ~954 GB of float embeddings to ~238 GB of int8 embeddings can cut storage requirements by around 75%, with only a minimal decrease in search quality (66.3 → 66.1).
- **Int8 vs. Binary:**  
  Going from int8 to binary reduces the memory footprint from ~238 GB to ~30 GB, leading to further cost reductions. Even though the search quality dips from 66.1 to 62.8, it may remain acceptable for certain use cases.
- **Cost Savings:**  
  These memory reductions translate to significant cost benefits. For instance, storing full-precision embeddings can cost \$43,488/year, while int8 embeddings might cost \$10,872/year, and binary embeddings only \$1,359/year.

These figures provide a real-world illustration of how int8 and binary quantization can reduce operational expenses while preserving most of the semantic content needed for high-quality search.

---

## CohereEnhancedVectorDB

The `CohereEnhancedVectorDB.py` module implements an enhanced vector database that leverages Cohere’s embedding API to generate multiple types of embeddings:

- **int8 Embeddings:**  
  Used for final cosine similarity rescoring.
  
- **ubinary Embeddings:**  
  Derived from int8 embeddings via thresholding and bit-packing, and used to build a fast FAISS binary index based on Hamming distance.
  
- **float Embeddings:**  
  Used for precise rescoring through both dot-product and cosine similarity measures.

This module employs a **three-phase search process**:

1. **Phase I:**  
   A rapid FAISS binary search retrieves an oversampled candidate set using the ubinary representation of the query.

2. **Phase II:**  
   Candidates are rescored using the dot-product between the query’s float embedding and the unpacked binary embeddings from the FAISS index.

3. **Phase III:**  
   The top candidates are further rescored using cosine similarity between the query’s float embedding and the stored int8 embeddings (retrieved from RocksDict).

By combining these multiple embedding types and search phases, `CohereEnhancedVectorDB` offers both fast retrieval and high precision, making it well-suited for large-scale document search applications.

---

## Benchmark Results and Analysis

### Float-Based Search (CohereFloat)
- **Build Time:** ~26.69 seconds
- **Search Time:** ~0.52 seconds
- **Index Size:** ~4.04 MB
- **Top Results (Example):**
  - DocID=851, Score ≈ 0.589586, Doc: "AI is transforming supply chain optimization."
  - DocID=659, Score ≈ 0.576121, Doc: "AI in sports is transforming manufacturing processes."
  - DocID=952, Score ≈ 0.574212, Doc: "AI in cybersecurity is revolutionizing manufacturing processes."

### Enhanced Multi-Phase Search (CohereEnhancedVectorDB)
- **Build Time:** ~14.45 seconds
- **Search Time:** ~0.51 seconds
- **Index Size:** ~1.38 MB
- **Top Results (Example):**
  - DocID=851, Score ≈ 0.588011, Doc: "AI is transforming supply chain optimization."
  - DocID=659, Score ≈ 0.575935, Doc: "AI in sports is transforming manufacturing processes."
  - DocID=952, Score ≈ 0.573149, Doc: "AI in cybersecurity is revolutionizing manufacturing processes."

### Comparison and Analysis

- **Result Accuracy:**  
  The top results produced by both methods are nearly identical, with score differences on the order of a few thousandths. This indicates that the enhanced multi-phase search accurately approximates the full-precision float search.

- **Efficiency Improvements:**  
  - **Build Time:** The enhanced method builds the index approximately 45% faster than the float-based approach.
  - **Index Size:** The enhanced approach produces an index that is roughly 66% smaller than that of the float-based method.
  - **Search Time:** Both methods have similar search times (~0.51–0.52 seconds).

- **Trade-Offs:**  
  While the float-based approach uses full-precision calculations, requiring more time and storage, the enhanced multi-phase search method achieves nearly identical ranking quality by combining fast binary search (via FAISS and ubinary embeddings) with efficient rescoring. This yields significant gains in build time and storage efficiency.

---

## Project Structure

```plaintext
.
├── main.py                     # Main script to build indexes, run searches, and benchmark performance
├── CohereEnhancedVectorDB.py   # Enhanced multi-phase search implementation
├── CohereVectorDBFloat.py      # Full-precision float vector database (if available)
├── CohereVectorDBInt8.py       # Cohere-specific int8 quantization-based vector database
├── CohereVectorDBBinary.py     # Signed binary quantization based vector database
├── VectorDBInt8.py             # Local per-document int8 quantization
├── VectorDBInt16.py            # Local per-document int16 quantization
├── VectorDBInt4.py             # Local per-document int4 quantization
├── VectorDBInt8Global.py       # Global-limit int8 quantization
├── VectorDBInt16Global.py      # Global-limit int16 quantization
├── VectorDBInt4Global.py       # Global-limit int4 quantization
├── embedding_service.py        # Example embedding service for float32 inference
├── Generated_AI_Examples.csv   # CSV file containing document texts
```

Environment Variables

Before running the project, ensure the following environment variables are set:
	•	COHERE_EMBED_ENDPOINT
The URL for the Cohere embedding API (e.g., https://api.cohere.ai/v2/embed).
	•	COHERE_EMBED_KEY
Your Cohere API key.
	•	(Optional for reranking) COHERE_RERANK_ENDPOINT and COHERE_RERANK_KEY
The endpoint and API key for Cohere’s rerank service.

For example, on Linux/Mac:


```bash

export COHERE_EMBED_ENDPOINT="https://api.cohere.ai/v2/embed"
export COHERE_EMBED_KEY="your_cohere_api_key_here"
export COHERE_RERANK_ENDPOINT="https://api.cohere.ai/v2/rerank"
export COHERE_RERANK_KEY="your_cohere_rerank_api_key_here"

```


## Usage
	1.	Prepare Your Data
The CSV file Generated_AI_Examples.csv should contain a column named Generated Examples with the document texts.
	2.	Run the Benchmark
Execute the main script:

```bash
python main.py
```

## Results

	•	Clean and create an output directory (img/) for plots.
	•	Load documents from the CSV.
	•	Build various vector databases (Float, Int8, Enhanced, etc.).
	•	Execute search queries using each database.
	•	Optionally perform reranking via Cohere’s API.
	•	Compare the performance and accuracy (build/search time, index sizes, score comparisons).
	•	Generate plots and save results to results.csv.

	3.	View Outputs
	•	Logs: Detailed logs are printed to the console.
	•	Plots: Images are stored in the img/ directory.
	•	CSV: The results.csv file contains detailed performance and score comparison data.



```log
2025-02-14 13:03:56,496 [INFO] Removing existing directory: img
2025-02-14 13:03:56,497 [INFO] Creating new directory: img
2025-02-14 13:03:56,503 [INFO] Loaded 1000 documents from Generated_AI_Examples.csv.
2025-02-14 13:03:56,503 [INFO] === Single-Stage Cohere Float (Benchmark) ===
2025-02-14 13:03:56,503 [INFO] Removing existing directory: ./db_cohere_float
2025-02-14 13:03:56,521 [INFO] New float FAISS index created (dim=1024).
2025-02-14 13:03:56,537 [INFO] Adding documents to Cohere Float DB...

Indexing docs (Float):   0%|          | 0/1000 [00:00<?, ?it/s]
Indexing docs (Float):   6%|▋         | 64/1000 [00:01<00:26, 34.91it/s]
Indexing docs (Float):  13%|█▎        | 128/1000 [00:03<00:23, 37.12it/s]
Indexing docs (Float):  19%|█▉        | 192/1000 [00:05<00:21, 37.01it/s]
Indexing docs (Float):  26%|██▌       | 256/1000 [00:06<00:19, 37.56it/s]
Indexing docs (Float):  32%|███▏      | 320/1000 [00:08<00:18, 37.46it/s]
Indexing docs (Float):  38%|███▊      | 384/1000 [00:10<00:16, 37.40it/s]
Indexing docs (Float):  45%|████▍     | 448/1000 [00:11<00:14, 37.89it/s]
Indexing docs (Float):  51%|█████     | 512/1000 [00:13<00:12, 37.83it/s]
Indexing docs (Float):  58%|█████▊    | 576/1000 [00:15<00:11, 38.08it/s]
Indexing docs (Float):  64%|██████▍   | 640/1000 [00:16<00:09, 38.14it/s]
Indexing docs (Float):  70%|███████   | 704/1000 [00:18<00:07, 37.16it/s]
Indexing docs (Float):  77%|███████▋  | 768/1000 [00:20<00:06, 37.54it/s]
Indexing docs (Float):  83%|████████▎ | 832/1000 [00:22<00:04, 38.41it/s]
Indexing docs (Float):  90%|████████▉ | 896/1000 [00:23<00:02, 38.89it/s]
Indexing docs (Float):  96%|█████████▌| 960/1000 [00:25<00:01, 38.48it/s]
Indexing docs (Float): 100%|██████████| 1000/1000 [00:26<00:00, 36.73it/s]
Indexing docs (Float): 100%|██████████| 1000/1000 [00:26<00:00, 37.54it/s]
2025-02-14 13:04:23,223 [INFO] Float FAISS index saved to disk.
2025-02-14 13:04:23,224 [INFO] Time to build Cohere Float DB: 26.69 seconds
2025-02-14 13:04:23,227 [INFO] Cohere Float DB size: 4239961 bytes
2025-02-14 13:04:23,228 [INFO] Performing float-based search for the query...
2025-02-14 13:04:23,751 [INFO] Time to retrieve Cohere Float results: 0.52 seconds
2025-02-14 13:04:23,751 [INFO] Search Results (CohereFloat):
2025-02-14 13:04:23,751 [INFO] QUERY: Artificial intelligence is transforming industries.
2025-02-14 13:04:23,751 [INFO]  DocID=851, Score=0.589586, Doc='AI is transforming supply chain optimization.'
2025-02-14 13:04:23,751 [INFO]  DocID=659, Score=0.576121, Doc='AI in sports is transforming manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=952, Score=0.574212, Doc='AI in cybersecurity is revolutionizing manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=548, Score=0.565028, Doc='AI is revolutionizing human-computer interaction.'
2025-02-14 13:04:23,751 [INFO]  DocID=242, Score=0.561637, Doc='AI is transforming cybersecurity systems.'
2025-02-14 13:04:23,751 [INFO]  DocID=620, Score=0.553172, Doc='AI in marketing is transforming fraud detection.'
2025-02-14 13:04:23,751 [INFO]  DocID=25, Score=0.551931, Doc='AI in marketing is improving manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=260, Score=0.551810, Doc='AI in healthcare is revolutionizing manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=643, Score=0.548874, Doc='Ethical AI is revolutionizing manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=510, Score=0.546837, Doc='AI is transforming smart city planning.'
2025-02-14 13:04:23,751 [INFO]  DocID=430, Score=0.544675, Doc='AI in marketing is transforming e-commerce engagement.'
2025-02-14 13:04:23,751 [INFO]  DocID=327, Score=0.541890, Doc='AI is transforming traffic management.'
2025-02-14 13:04:23,751 [INFO]  DocID=695, Score=0.541130, Doc='AI in education is revolutionizing manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=590, Score=0.537523, Doc='AI in logistics is transforming smart city planning.'
2025-02-14 13:04:23,751 [INFO]  DocID=703, Score=0.535620, Doc='AI-powered virtual assistants creates opportunities for manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=653, Score=0.532789, Doc='AI-powered virtual assistants is transforming e-commerce engagement.'
2025-02-14 13:04:23,751 [INFO]  DocID=124, Score=0.532756, Doc='AI in logistics is transforming fraud detection.'
2025-02-14 13:04:23,751 [INFO]  DocID=375, Score=0.532603, Doc='AI in logistics is improving manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=463, Score=0.529190, Doc='AI in marketing enhances manufacturing processes.'
2025-02-14 13:04:23,751 [INFO]  DocID=177, Score=0.528512, Doc='AI in marketing is transforming translation accuracy.'
2025-02-14 13:04:23,751 [INFO]  DocID=827, Score=0.527312, Doc='Explainable AI is transforming e-commerce engagement.'
2025-02-14 13:04:23,751 [INFO]  DocID=485, Score=0.523139, Doc='Quantum computing is transforming artificial intelligence research.'
2025-02-14 13:04:23,751 [INFO]  DocID=597, Score=0.522445, Doc='AI in healthcare is revolutionizing supply chain optimization.'
2025-02-14 13:04:23,751 [INFO]  DocID=240, Score=0.520930, Doc='AI is transforming player performance in sports.'
2025-02-14 13:04:23,751 [INFO]  DocID=864, Score=0.519565, Doc='Self-driving cars drives advancements in artificial intelligence research.'
2025-02-14 13:04:23,751 [INFO]  DocID=52, Score=0.519240, Doc='AI in marketing is transforming astronomical data analysis.'
2025-02-14 13:04:23,751 [INFO]  DocID=434, Score=0.517842, Doc='AI in marketing is transforming player performance in sports.'
2025-02-14 13:04:23,752 [INFO]  DocID=871, Score=0.517551, Doc='Smart cities is transforming artificial intelligence research.'
2025-02-14 13:04:23,752 [INFO]  DocID=47, Score=0.515186, Doc='AI-powered fraud detection is transforming e-commerce engagement.'
2025-02-14 13:04:23,752 [INFO]  DocID=874, Score=0.514357, Doc='AI-powered fraud detection is improving manufacturing processes.'
2025-02-14 13:04:23,752 [INFO]  DocID=499, Score=0.514206, Doc='AI streamlines human-computer interaction.'
2025-02-14 13:04:23,752 [INFO]  DocID=525, Score=0.510798, Doc='Explainable AI is transforming smart city planning.'
2025-02-14 13:04:23,752 [INFO]  DocID=839, Score=0.509582, Doc='AI in logistics is transforming education personalization.'
2025-02-14 13:04:23,752 [INFO]  DocID=381, Score=0.509481, Doc='AI-powered virtual assistants personalizes manufacturing processes.'
2025-02-14 13:04:23,752 [INFO]  DocID=440, Score=0.507551, Doc='AI in sports is revolutionizing supply chain optimization.'
2025-02-14 13:04:23,752 [INFO]  DocID=627, Score=0.505557, Doc='Generative AI is revolutionizing artificial intelligence research.'
2025-02-14 13:04:23,752 [INFO]  DocID=670, Score=0.504622, Doc='AI in logistics is revolutionizing customer experience.'
2025-02-14 13:04:23,752 [INFO]  DocID=208, Score=0.504317, Doc='AI drives advancements in fraud detection.'
2025-02-14 13:04:23,752 [INFO]  DocID=521, Score=0.503942, Doc='AI-powered fraud detection is revolutionizing logistics efficiency.'
2025-02-14 13:04:23,752 [INFO]  DocID=503, Score=0.502571, Doc='AI in healthcare is revolutionizing logistics efficiency.'
2025-02-14 13:04:23,752 [INFO]  DocID=219, Score=0.501247, Doc='Machine learning is transforming supply chain optimization.'
2025-02-14 13:04:23,752 [INFO]  DocID=675, Score=0.500813, Doc='AI-powered virtual assistants drives advancements in customer experience.'
2025-02-14 13:04:23,752 [INFO]  DocID=832, Score=0.497726, Doc='AI is revolutionizing document review.'
2025-02-14 13:04:23,752 [INFO]  DocID=858, Score=0.496725, Doc='AI is improving e-commerce engagement.'
2025-02-14 13:04:23,752 [INFO]  DocID=282, Score=0.496130, Doc='AI in climate science is transforming financial risk management.'
2025-02-14 13:04:23,752 [INFO]  DocID=791, Score=0.496105, Doc='Explainable AI is revolutionizing e-commerce engagement.'
2025-02-14 13:04:23,752 [INFO]  DocID=110, Score=0.494978, Doc='AI in marketing is revolutionizing crop yields.'
2025-02-14 13:04:23,752 [INFO]  DocID=822, Score=0.494444, Doc='AI-powered fraud detection is reshaping customer experience.'
2025-02-14 13:04:23,752 [INFO]  DocID=145, Score=0.493950, Doc='AI in climate science is improving manufacturing processes.'
2025-02-14 13:04:23,752 [INFO]  DocID=581, Score=0.492573, Doc='AI in finance is transforming crop yields.'
2025-02-14 13:04:23,768 [INFO] Results saved for CohereFloat to results.csv
2025-02-14 13:04:23,782 [INFO] === Single-Stage Cohere Int8 (Local) ===
2025-02-14 13:04:23,782 [INFO] Removing existing directory: ./db_cohere_int8
2025-02-14 13:04:23,801 [INFO] New FAISS binary index created with embedding dimension 1024.
2025-02-14 13:04:23,805 [INFO] Adding documents to Cohere Int8 DB...

Indexing docs (Int8):   0%|          | 0/1000 [00:00<?, ?it/s]
Indexing docs (Int8):   6%|▋         | 64/1000 [00:00<00:13, 67.91it/s]
Indexing docs (Int8):  13%|█▎        | 128/1000 [00:01<00:12, 69.76it/s]
Indexing docs (Int8):  19%|█▉        | 192/1000 [00:02<00:11, 69.22it/s]
Indexing docs (Int8):  26%|██▌       | 256/1000 [00:03<00:10, 70.31it/s]
Indexing docs (Int8):  32%|███▏      | 320/1000 [00:04<00:09, 70.35it/s]
Indexing docs (Int8):  38%|███▊      | 384/1000 [00:05<00:08, 70.22it/s]
Indexing docs (Int8):  45%|████▍     | 448/1000 [00:06<00:07, 70.08it/s]
Indexing docs (Int8):  51%|█████     | 512/1000 [00:07<00:06, 70.69it/s]
Indexing docs (Int8):  58%|█████▊    | 576/1000 [00:08<00:06, 70.45it/s]
Indexing docs (Int8):  64%|██████▍   | 640/1000 [00:09<00:05, 70.81it/s]
Indexing docs (Int8):  70%|███████   | 704/1000 [00:10<00:04, 70.92it/s]
Indexing docs (Int8):  77%|███████▋  | 768/1000 [00:10<00:03, 70.23it/s]
Indexing docs (Int8):  83%|████████▎ | 832/1000 [00:11<00:02, 69.95it/s]
Indexing docs (Int8):  90%|████████▉ | 896/1000 [00:12<00:01, 70.19it/s]
Indexing docs (Int8):  96%|█████████▌| 960/1000 [00:13<00:00, 70.30it/s]
Indexing docs (Int8): 100%|██████████| 1000/1000 [00:14<00:00, 64.66it/s]
Indexing docs (Int8): 100%|██████████| 1000/1000 [00:14<00:00, 69.11it/s]
2025-02-14 13:04:38,345 [INFO] FAISS int8 binary index saved to disk.
2025-02-14 13:04:38,345 [INFO] Time to build Cohere Int8 DB: 14.54 seconds
2025-02-14 13:04:38,346 [INFO] Cohere Int8 DB size: 1444217 bytes
2025-02-14 13:04:38,855 [INFO] Time to retrieve raw Cohere Int8 results: 0.51 seconds
2025-02-14 13:04:38,857 [INFO] Raw Search Results (Cohere Int8):
2025-02-14 13:04:38,857 [INFO] QUERY: Artificial intelligence is transforming industries.
2025-02-14 13:04:38,857 [INFO]  DocID=620, Raw Score=292, Doc='AI in marketing is transforming fraud detection.'
2025-02-14 13:04:38,857 [INFO]  DocID=851, Raw Score=293, Doc='AI is transforming supply chain optimization.'
2025-02-14 13:04:38,857 [INFO]  DocID=695, Raw Score=298, Doc='AI in education is revolutionizing manufacturing processes.'
2025-02-14 13:04:38,857 [INFO]  DocID=124, Raw Score=300, Doc='AI in logistics is transforming fraud detection.'
2025-02-14 13:04:38,857 [INFO]  DocID=485, Raw Score=300, Doc='Quantum computing is transforming artificial intelligence research.'
2025-02-14 13:04:38,857 [INFO]  DocID=703, Raw Score=300, Doc='AI-powered virtual assistants creates opportunities for manufacturing processes.'
2025-02-14 13:04:38,857 [INFO]  DocID=430, Raw Score=301, Doc='AI in marketing is transforming e-commerce engagement.'
2025-02-14 13:04:38,857 [INFO]  DocID=952, Raw Score=301, Doc='AI in cybersecurity is revolutionizing manufacturing processes.'
2025-02-14 13:04:38,857 [INFO]  DocID=510, Raw Score=302, Doc='AI is transforming smart city planning.'
2025-02-14 13:04:38,858 [INFO]  DocID=434, Raw Score=303, Doc='AI in marketing is transforming player performance in sports.'
2025-02-14 13:04:38,858 [INFO]  DocID=659, Raw Score=304, Doc='AI in sports is transforming manufacturing processes.'
2025-02-14 13:04:38,858 [INFO]  DocID=282, Raw Score=306, Doc='AI in climate science is transforming financial risk management.'
2025-02-14 13:04:38,858 [INFO]  DocID=548, Raw Score=306, Doc='AI is revolutionizing human-computer interaction.'
2025-02-14 13:04:38,858 [INFO]  DocID=864, Raw Score=306, Doc='Self-driving cars drives advancements in artificial intelligence research.'
2025-02-14 13:04:38,858 [INFO]  DocID=208, Raw Score=308, Doc='AI drives advancements in fraud detection.'
2025-02-14 13:04:38,858 [INFO]  DocID=871, Raw Score=308, Doc='Smart cities is transforming artificial intelligence research.'
2025-02-14 13:04:38,858 [INFO]  DocID=590, Raw Score=309, Doc='AI in logistics is transforming smart city planning.'
2025-02-14 13:04:38,858 [INFO]  DocID=597, Raw Score=309, Doc='AI in healthcare is revolutionizing supply chain optimization.'
2025-02-14 13:04:38,858 [INFO]  DocID=764, Raw Score=309, Doc='AI in agriculture is transforming education personalization.'
2025-02-14 13:04:38,858 [INFO]  DocID=827, Raw Score=309, Doc='Explainable AI is transforming e-commerce engagement.'
2025-02-14 13:04:38,858 [INFO]  DocID=25, Raw Score=310, Doc='AI in marketing is improving manufacturing processes.'
2025-02-14 13:04:38,858 [INFO]  DocID=242, Raw Score=310, Doc='AI is transforming cybersecurity systems.'
2025-02-14 13:04:38,858 [INFO]  DocID=260, Raw Score=310, Doc='AI in healthcare is revolutionizing manufacturing processes.'
2025-02-14 13:04:38,858 [INFO]  DocID=463, Raw Score=310, Doc='AI in marketing enhances manufacturing processes.'
2025-02-14 13:04:38,858 [INFO]  DocID=375, Raw Score=312, Doc='AI in logistics is improving manufacturing processes.'
2025-02-14 13:04:38,858 [INFO]  DocID=643, Raw Score=313, Doc='Ethical AI is revolutionizing manufacturing processes.'
2025-02-14 13:04:38,858 [INFO]  DocID=30, Raw Score=314, Doc='AI in the legal field is reshaping artificial intelligence research.'
2025-02-14 13:04:38,858 [INFO]  DocID=52, Raw Score=314, Doc='AI in marketing is transforming astronomical data analysis.'
2025-02-14 13:04:38,858 [INFO]  DocID=377, Raw Score=314, Doc='AI in cybersecurity is transforming real estate valuation.'
2025-02-14 13:04:38,859 [INFO]  DocID=581, Raw Score=314, Doc='AI in finance is transforming crop yields.'
2025-02-14 13:04:38,859 [INFO]  DocID=177, Raw Score=315, Doc='AI in marketing is transforming translation accuracy.'
2025-02-14 13:04:38,859 [INFO]  DocID=839, Raw Score=315, Doc='AI in logistics is transforming education personalization.'
2025-02-14 13:04:38,859 [INFO]  DocID=219, Raw Score=316, Doc='Machine learning is transforming supply chain optimization.'
2025-02-14 13:04:38,859 [INFO]  DocID=440, Raw Score=316, Doc='AI in sports is revolutionizing supply chain optimization.'
2025-02-14 13:04:38,859 [INFO]  DocID=612, Raw Score=316, Doc='Explainable AI drives advancements in fraud detection.'
2025-02-14 13:04:38,859 [INFO]  DocID=791, Raw Score=316, Doc='Explainable AI is revolutionizing e-commerce engagement.'
2025-02-14 13:04:38,859 [INFO]  DocID=734, Raw Score=317, Doc='AI in finance is revolutionizing healthcare outcomes.'
2025-02-14 13:04:38,859 [INFO]  DocID=832, Raw Score=317, Doc='AI is revolutionizing document review.'
2025-02-14 13:04:38,859 [INFO]  DocID=653, Raw Score=318, Doc='AI-powered virtual assistants is transforming e-commerce engagement.'
2025-02-14 13:04:38,859 [INFO]  DocID=47, Raw Score=320, Doc='AI-powered fraud detection is transforming e-commerce engagement.'
2025-02-14 13:04:38,859 [INFO]  DocID=240, Raw Score=320, Doc='AI is transforming player performance in sports.'
2025-02-14 13:04:38,859 [INFO]  DocID=604, Raw Score=320, Doc='AI in education drives advancements in e-commerce engagement.'
2025-02-14 13:04:38,859 [INFO]  DocID=661, Raw Score=320, Doc='AI in cybersecurity is reshaping financial risk management.'
2025-02-14 13:04:38,859 [INFO]  DocID=197, Raw Score=321, Doc='Explainable AI drives advancements in document review.'
2025-02-14 13:04:38,859 [INFO]  DocID=309, Raw Score=321, Doc='AI in space exploration is transforming customer experience.'
2025-02-14 13:04:38,859 [INFO]  DocID=473, Raw Score=321, Doc='AI in education drives advancements in human-computer interaction.'
2025-02-14 13:04:38,859 [INFO]  DocID=991, Raw Score=321, Doc='AI in healthcare is reshaping energy efficiency.'
2025-02-14 13:04:38,860 [INFO]  DocID=381, Raw Score=322, Doc='AI-powered virtual assistants personalizes manufacturing processes.'
2025-02-14 13:04:38,860 [INFO]  DocID=429, Raw Score=322, Doc='AI in finance creates opportunities for e-commerce engagement.'
2025-02-14 13:04:38,860 [INFO]  DocID=28, Raw Score=323, Doc='AI in the legal field is transforming crop yields.'
2025-02-14 13:04:38,866 [INFO] Existing float FAISS index loaded.
2025-02-14 13:04:39,384 [INFO] Target range from CohereFloat: min=0.492573, max=0.589586
2025-02-14 13:04:39,384 [INFO] Normalized Search Results (Cohere Int8):
2025-02-14 13:04:39,384 [INFO]  DocID=28, Normalized Score=0.589586, Doc='AI in the legal field is transforming crop yields.'
2025-02-14 13:04:39,384 [INFO]  DocID=381, Normalized Score=0.586457, Doc='AI-powered virtual assistants personalizes manufacturing processes.'
2025-02-14 13:04:39,384 [INFO]  DocID=429, Normalized Score=0.586457, Doc='AI in finance creates opportunities for e-commerce engagement.'
2025-02-14 13:04:39,384 [INFO]  DocID=197, Normalized Score=0.583328, Doc='Explainable AI drives advancements in document review.'
2025-02-14 13:04:39,384 [INFO]  DocID=309, Normalized Score=0.583328, Doc='AI in space exploration is transforming customer experience.'
2025-02-14 13:04:39,385 [INFO]  DocID=473, Normalized Score=0.583328, Doc='AI in education drives advancements in human-computer interaction.'
2025-02-14 13:04:39,385 [INFO]  DocID=991, Normalized Score=0.583328, Doc='AI in healthcare is reshaping energy efficiency.'
2025-02-14 13:04:39,385 [INFO]  DocID=47, Normalized Score=0.580198, Doc='AI-powered fraud detection is transforming e-commerce engagement.'
2025-02-14 13:04:39,385 [INFO]  DocID=240, Normalized Score=0.580198, Doc='AI is transforming player performance in sports.'
2025-02-14 13:04:39,385 [INFO]  DocID=604, Normalized Score=0.580198, Doc='AI in education drives advancements in e-commerce engagement.'
2025-02-14 13:04:39,385 [INFO]  DocID=661, Normalized Score=0.580198, Doc='AI in cybersecurity is reshaping financial risk management.'
2025-02-14 13:04:39,385 [INFO]  DocID=653, Normalized Score=0.573939, Doc='AI-powered virtual assistants is transforming e-commerce engagement.'
2025-02-14 13:04:39,385 [INFO]  DocID=734, Normalized Score=0.570810, Doc='AI in finance is revolutionizing healthcare outcomes.'
2025-02-14 13:04:39,385 [INFO]  DocID=832, Normalized Score=0.570810, Doc='AI is revolutionizing document review.'
2025-02-14 13:04:39,385 [INFO]  DocID=219, Normalized Score=0.567680, Doc='Machine learning is transforming supply chain optimization.'
2025-02-14 13:04:39,385 [INFO]  DocID=440, Normalized Score=0.567680, Doc='AI in sports is revolutionizing supply chain optimization.'
2025-02-14 13:04:39,385 [INFO]  DocID=612, Normalized Score=0.567680, Doc='Explainable AI drives advancements in fraud detection.'
2025-02-14 13:04:39,385 [INFO]  DocID=791, Normalized Score=0.567680, Doc='Explainable AI is revolutionizing e-commerce engagement.'
2025-02-14 13:04:39,385 [INFO]  DocID=177, Normalized Score=0.564551, Doc='AI in marketing is transforming translation accuracy.'
2025-02-14 13:04:39,385 [INFO]  DocID=839, Normalized Score=0.564551, Doc='AI in logistics is transforming education personalization.'
2025-02-14 13:04:39,385 [INFO]  DocID=30, Normalized Score=0.561421, Doc='AI in the legal field is reshaping artificial intelligence research.'
2025-02-14 13:04:39,385 [INFO]  DocID=52, Normalized Score=0.561421, Doc='AI in marketing is transforming astronomical data analysis.'
2025-02-14 13:04:39,385 [INFO]  DocID=377, Normalized Score=0.561421, Doc='AI in cybersecurity is transforming real estate valuation.'
2025-02-14 13:04:39,385 [INFO]  DocID=581, Normalized Score=0.561421, Doc='AI in finance is transforming crop yields.'
2025-02-14 13:04:39,385 [INFO]  DocID=643, Normalized Score=0.558292, Doc='Ethical AI is revolutionizing manufacturing processes.'
2025-02-14 13:04:39,385 [INFO]  DocID=375, Normalized Score=0.555162, Doc='AI in logistics is improving manufacturing processes.'
2025-02-14 13:04:39,385 [INFO]  DocID=25, Normalized Score=0.548903, Doc='AI in marketing is improving manufacturing processes.'
2025-02-14 13:04:39,385 [INFO]  DocID=242, Normalized Score=0.548903, Doc='AI is transforming cybersecurity systems.'
2025-02-14 13:04:39,385 [INFO]  DocID=260, Normalized Score=0.548903, Doc='AI in healthcare is revolutionizing manufacturing processes.'
2025-02-14 13:04:39,385 [INFO]  DocID=463, Normalized Score=0.548903, Doc='AI in marketing enhances manufacturing processes.'
2025-02-14 13:04:39,385 [INFO]  DocID=590, Normalized Score=0.545774, Doc='AI in logistics is transforming smart city planning.'
2025-02-14 13:04:39,385 [INFO]  DocID=597, Normalized Score=0.545774, Doc='AI in healthcare is revolutionizing supply chain optimization.'
2025-02-14 13:04:39,385 [INFO]  DocID=764, Normalized Score=0.545774, Doc='AI in agriculture is transforming education personalization.'
2025-02-14 13:04:39,385 [INFO]  DocID=827, Normalized Score=0.545774, Doc='Explainable AI is transforming e-commerce engagement.'
2025-02-14 13:04:39,385 [INFO]  DocID=208, Normalized Score=0.542644, Doc='AI drives advancements in fraud detection.'
2025-02-14 13:04:39,385 [INFO]  DocID=871, Normalized Score=0.542644, Doc='Smart cities is transforming artificial intelligence research.'
2025-02-14 13:04:39,385 [INFO]  DocID=282, Normalized Score=0.536385, Doc='AI in climate science is transforming financial risk management.'
2025-02-14 13:04:39,386 [INFO]  DocID=548, Normalized Score=0.536385, Doc='AI is revolutionizing human-computer interaction.'
2025-02-14 13:04:39,386 [INFO]  DocID=864, Normalized Score=0.536385, Doc='Self-driving cars drives advancements in artificial intelligence research.'
2025-02-14 13:04:39,386 [INFO]  DocID=659, Normalized Score=0.530126, Doc='AI in sports is transforming manufacturing processes.'
2025-02-14 13:04:39,386 [INFO]  DocID=434, Normalized Score=0.526997, Doc='AI in marketing is transforming player performance in sports.'
2025-02-14 13:04:39,386 [INFO]  DocID=510, Normalized Score=0.523867, Doc='AI is transforming smart city planning.'
2025-02-14 13:04:39,386 [INFO]  DocID=430, Normalized Score=0.520738, Doc='AI in marketing is transforming e-commerce engagement.'
2025-02-14 13:04:39,386 [INFO]  DocID=952, Normalized Score=0.520738, Doc='AI in cybersecurity is revolutionizing manufacturing processes.'
2025-02-14 13:04:39,386 [INFO]  DocID=124, Normalized Score=0.517608, Doc='AI in logistics is transforming fraud detection.'
2025-02-14 13:04:39,386 [INFO]  DocID=485, Normalized Score=0.517608, Doc='Quantum computing is transforming artificial intelligence research.'
2025-02-14 13:04:39,386 [INFO]  DocID=703, Normalized Score=0.517608, Doc='AI-powered virtual assistants creates opportunities for manufacturing processes.'
2025-02-14 13:04:39,386 [INFO]  DocID=695, Normalized Score=0.511350, Doc='AI in education is revolutionizing manufacturing processes.'
2025-02-14 13:04:39,386 [INFO]  DocID=851, Normalized Score=0.495702, Doc='AI is transforming supply chain optimization.'
2025-02-14 13:04:39,386 [INFO]  DocID=620, Normalized Score=0.492573, Doc='AI in marketing is transforming fraud detection.'
2025-02-14 13:04:39,399 [INFO] Results saved for CohereInt8 to results.csv
2025-02-14 13:04:39,405 [INFO] === Enhanced Cohere DB (Multi-Phase: int8/ubinary/float) ===
2025-02-14 13:04:39,405 [INFO] Removing existing directory: ./db_cohere_enhanced
2025-02-14 13:04:39,445 [INFO] New FAISS binary index created with embedding dimension 1024.
2025-02-14 13:04:39,449 [INFO] Adding documents to Cohere Enhanced DB...

Indexing documents:   0%|          | 0/1000 [00:00<?, ?it/s]
Indexing documents:   6%|▋         | 64/1000 [00:00<00:13, 69.25it/s]
Indexing documents:  13%|█▎        | 128/1000 [00:01<00:12, 68.60it/s]
Indexing documents:  19%|█▉        | 192/1000 [00:02<00:11, 69.73it/s]
Indexing documents:  26%|██▌       | 256/1000 [00:03<00:10, 69.11it/s]
Indexing documents:  32%|███▏      | 320/1000 [00:04<00:09, 69.57it/s]
Indexing documents:  38%|███▊      | 384/1000 [00:05<00:08, 69.09it/s]
Indexing documents:  45%|████▍     | 448/1000 [00:06<00:07, 69.74it/s]
Indexing documents:  51%|█████     | 512/1000 [00:07<00:06, 70.09it/s]
Indexing documents:  58%|█████▊    | 576/1000 [00:08<00:05, 70.72it/s]
Indexing documents:  64%|██████▍   | 640/1000 [00:09<00:05, 70.62it/s]
Indexing documents:  70%|███████   | 704/1000 [00:10<00:04, 71.24it/s]
Indexing documents:  77%|███████▋  | 768/1000 [00:10<00:03, 71.15it/s]
Indexing documents:  83%|████████▎ | 832/1000 [00:11<00:02, 70.57it/s]
Indexing documents:  90%|████████▉ | 896/1000 [00:12<00:01, 70.82it/s]
Indexing documents:  96%|█████████▌| 960/1000 [00:13<00:00, 71.09it/s]
Indexing documents: 100%|██████████| 1000/1000 [00:14<00:00, 65.18it/s]
Indexing documents: 100%|██████████| 1000/1000 [00:14<00:00, 69.21it/s]
2025-02-14 13:04:53,898 [INFO] FAISS binary index saved.
2025-02-14 13:04:53,899 [INFO] Time to build Cohere Enhanced DB: 14.45 seconds
2025-02-14 13:04:53,899 [INFO] Cohere Enhanced DB size: 1444232 bytes
2025-02-14 13:04:53,899 [INFO] Performing enhanced search for the query...
2025-02-14 13:04:54,402 [INFO] Phase II rescoring (binary dot-product) took 7.94 ms
2025-02-14 13:04:54,404 [INFO] Phase III rescoring (cosine similarity) took 1.72 ms
2025-02-14 13:04:54,404 [INFO] Time to retrieve enhanced results: 0.51 seconds
2025-02-14 13:04:54,404 [INFO] Enhanced Search Results:
2025-02-14 13:04:54,405 [INFO] QUERY: Artificial intelligence is transforming industries.
2025-02-14 13:04:54,405 [INFO]  DocID=851, Score=0.588011, Doc='AI is transforming supply chain optimization.'
2025-02-14 13:04:54,405 [INFO]  DocID=659, Score=0.575935, Doc='AI in sports is transforming manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=952, Score=0.573149, Doc='AI in cybersecurity is revolutionizing manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=548, Score=0.564410, Doc='AI is revolutionizing human-computer interaction.'
2025-02-14 13:04:54,405 [INFO]  DocID=242, Score=0.561112, Doc='AI is transforming cybersecurity systems.'
2025-02-14 13:04:54,405 [INFO]  DocID=620, Score=0.552013, Doc='AI in marketing is transforming fraud detection.'
2025-02-14 13:04:54,405 [INFO]  DocID=25, Score=0.551568, Doc='AI in marketing is improving manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=260, Score=0.550459, Doc='AI in healthcare is revolutionizing manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=643, Score=0.548444, Doc='Ethical AI is revolutionizing manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=510, Score=0.546443, Doc='AI is transforming smart city planning.'
2025-02-14 13:04:54,405 [INFO]  DocID=430, Score=0.543895, Doc='AI in marketing is transforming e-commerce engagement.'
2025-02-14 13:04:54,405 [INFO]  DocID=327, Score=0.541100, Doc='AI is transforming traffic management.'
2025-02-14 13:04:54,405 [INFO]  DocID=695, Score=0.540202, Doc='AI in education is revolutionizing manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=590, Score=0.536036, Doc='AI in logistics is transforming smart city planning.'
2025-02-14 13:04:54,405 [INFO]  DocID=703, Score=0.535690, Doc='AI-powered virtual assistants creates opportunities for manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=653, Score=0.532063, Doc='AI-powered virtual assistants is transforming e-commerce engagement.'
2025-02-14 13:04:54,405 [INFO]  DocID=375, Score=0.532061, Doc='AI in logistics is improving manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=124, Score=0.530685, Doc='AI in logistics is transforming fraud detection.'
2025-02-14 13:04:54,405 [INFO]  DocID=463, Score=0.528162, Doc='AI in marketing enhances manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=177, Score=0.527359, Doc='AI in marketing is transforming translation accuracy.'
2025-02-14 13:04:54,405 [INFO]  DocID=827, Score=0.526420, Doc='Explainable AI is transforming e-commerce engagement.'
2025-02-14 13:04:54,405 [INFO]  DocID=485, Score=0.522248, Doc='Quantum computing is transforming artificial intelligence research.'
2025-02-14 13:04:54,405 [INFO]  DocID=240, Score=0.520587, Doc='AI is transforming player performance in sports.'
2025-02-14 13:04:54,405 [INFO]  DocID=597, Score=0.520462, Doc='AI in healthcare is revolutionizing supply chain optimization.'
2025-02-14 13:04:54,405 [INFO]  DocID=52, Score=0.518330, Doc='AI in marketing is transforming astronomical data analysis.'
2025-02-14 13:04:54,405 [INFO]  DocID=864, Score=0.518180, Doc='Self-driving cars drives advancements in artificial intelligence research.'
2025-02-14 13:04:54,405 [INFO]  DocID=434, Score=0.517153, Doc='AI in marketing is transforming player performance in sports.'
2025-02-14 13:04:54,405 [INFO]  DocID=871, Score=0.516343, Doc='Smart cities is transforming artificial intelligence research.'
2025-02-14 13:04:54,405 [INFO]  DocID=47, Score=0.514544, Doc='AI-powered fraud detection is transforming e-commerce engagement.'
2025-02-14 13:04:54,405 [INFO]  DocID=874, Score=0.513903, Doc='AI-powered fraud detection is improving manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=499, Score=0.513830, Doc='AI streamlines human-computer interaction.'
2025-02-14 13:04:54,405 [INFO]  DocID=525, Score=0.509896, Doc='Explainable AI is transforming smart city planning.'
2025-02-14 13:04:54,405 [INFO]  DocID=381, Score=0.509551, Doc='AI-powered virtual assistants personalizes manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=839, Score=0.507009, Doc='AI in logistics is transforming education personalization.'
2025-02-14 13:04:54,405 [INFO]  DocID=440, Score=0.505765, Doc='AI in sports is revolutionizing supply chain optimization.'
2025-02-14 13:04:54,405 [INFO]  DocID=627, Score=0.505248, Doc='Generative AI is revolutionizing artificial intelligence research.'
2025-02-14 13:04:54,405 [INFO]  DocID=208, Score=0.503145, Doc='AI drives advancements in fraud detection.'
2025-02-14 13:04:54,405 [INFO]  DocID=521, Score=0.502180, Doc='AI-powered fraud detection is revolutionizing logistics efficiency.'
2025-02-14 13:04:54,405 [INFO]  DocID=670, Score=0.501692, Doc='AI in logistics is revolutionizing customer experience.'
2025-02-14 13:04:54,405 [INFO]  DocID=503, Score=0.501032, Doc='AI in healthcare is revolutionizing logistics efficiency.'
2025-02-14 13:04:54,405 [INFO]  DocID=219, Score=0.499554, Doc='Machine learning is transforming supply chain optimization.'
2025-02-14 13:04:54,405 [INFO]  DocID=675, Score=0.499499, Doc='AI-powered virtual assistants drives advancements in customer experience.'
2025-02-14 13:04:54,405 [INFO]  DocID=832, Score=0.497559, Doc='AI is revolutionizing document review.'
2025-02-14 13:04:54,405 [INFO]  DocID=791, Score=0.495543, Doc='Explainable AI is revolutionizing e-commerce engagement.'
2025-02-14 13:04:54,405 [INFO]  DocID=858, Score=0.495376, Doc='AI is improving e-commerce engagement.'
2025-02-14 13:04:54,405 [INFO]  DocID=282, Score=0.494560, Doc='AI in climate science is transforming financial risk management.'
2025-02-14 13:04:54,405 [INFO]  DocID=145, Score=0.493891, Doc='AI in climate science is improving manufacturing processes.'
2025-02-14 13:04:54,405 [INFO]  DocID=110, Score=0.493783, Doc='AI in marketing is revolutionizing crop yields.'
2025-02-14 13:04:54,405 [INFO]  DocID=822, Score=0.491794, Doc='AI-powered fraud detection is reshaping customer experience.'
2025-02-14 13:04:54,405 [INFO]  DocID=269, Score=0.491438, Doc='Self-driving cars is revolutionizing manufacturing processes.'
2025-02-14 13:04:54,417 [INFO] Results saved for CohereEnhanced to results.csv
2025-02-14 13:04:54,421 [INFO] Existing FAISS binary index loaded.
2025-02-14 13:04:54,426 [INFO] Calling search_rerank_cohere()...
2025-02-14 13:04:54,939 [INFO] Calling Cohere rerank API at https://aitcohere-rerank-v3-english.eastus.models.ai.azure.com/v2/rerank
2025-02-14 13:04:54,940 [INFO] Payload for rerank: {'model': 'rerank-english-v3.0', 'query': 'Artificial intelligence is transforming industries.', 'top_n': 50, 'documents': ['AI in marketing is transforming fraud detection.', 'AI is transforming supply chain optimization.', 'AI in education is revolutionizing manufacturing processes.', 'AI in logistics is transforming fraud detection.', 'Quantum computing is transforming artificial intelligence research.', 'AI-powered virtual assistants creates opportunities for manufacturing processes.', 'AI in marketing is transforming e-commerce engagement.', 'AI in cybersecurity is revolutionizing manufacturing processes.', 'AI is transforming smart city planning.', 'AI in marketing is transforming player performance in sports.', 'AI in sports is transforming manufacturing processes.', 'AI in climate science is transforming financial risk management.', 'AI is revolutionizing human-computer interaction.', 'Self-driving cars drives advancements in artificial intelligence research.', 'AI drives advancements in fraud detection.', 'Smart cities is transforming artificial intelligence research.', 'AI in logistics is transforming smart city planning.', 'AI in healthcare is revolutionizing supply chain optimization.', 'AI in agriculture is transforming education personalization.', 'Explainable AI is transforming e-commerce engagement.', 'AI in marketing is improving manufacturing processes.', 'AI is transforming cybersecurity systems.', 'AI in healthcare is revolutionizing manufacturing processes.', 'AI in marketing enhances manufacturing processes.', 'AI in logistics is improving manufacturing processes.', 'Ethical AI is revolutionizing manufacturing processes.', 'AI in the legal field is reshaping artificial intelligence research.', 'AI in marketing is transforming astronomical data analysis.', 'AI in cybersecurity is transforming real estate valuation.', 'AI in finance is transforming crop yields.', 'AI in marketing is transforming translation accuracy.', 'AI in logistics is transforming education personalization.', 'Machine learning is transforming supply chain optimization.', 'AI in sports is revolutionizing supply chain optimization.', 'Explainable AI drives advancements in fraud detection.', 'Explainable AI is revolutionizing e-commerce engagement.', 'AI in finance is revolutionizing healthcare outcomes.', 'AI is revolutionizing document review.', 'AI-powered virtual assistants is transforming e-commerce engagement.', 'AI-powered fraud detection is transforming e-commerce engagement.', 'AI is transforming player performance in sports.', 'AI in education drives advancements in e-commerce engagement.', 'AI in cybersecurity is reshaping financial risk management.', 'Explainable AI drives advancements in document review.', 'AI in space exploration is transforming customer experience.', 'AI in education drives advancements in human-computer interaction.', 'AI in healthcare is reshaping energy efficiency.', 'AI-powered virtual assistants personalizes manufacturing processes.', 'AI in finance creates opportunities for e-commerce engagement.', 'AI in the legal field is transforming crop yields.', 'AI in the legal field is revolutionizing fraud detection.', 'Generative AI is revolutionizing artificial intelligence research.', 'AI is improving e-commerce engagement.', 'Ethical AI is revolutionizing fraud detection.', 'AI in climate science is transforming education personalization.', 'AI in healthcare is revolutionizing logistics efficiency.', 'AI-powered fraud detection is reshaping customer experience.', 'AI-powered fraud detection is improving manufacturing processes.', 'AI in cybersecurity is reshaping education personalization.', 'AI in cybersecurity is reshaping education personalization.', 'AI-powered virtual assistants drives advancements in customer experience.', 'AI is transforming traffic management.', 'AI in finance is revolutionizing climate science predictions.', 'AI in sports is revolutionizing real estate valuation.', 'AI aids in e-commerce engagement.', 'Ethical AI is reshaping logistics efficiency.', 'AI in climate science is reshaping e-commerce engagement.', 'AI in healthcare drives advancements in human-computer interaction.', 'AI in sports drives advancements in drug discovery.', 'Quantum computing accelerates artificial intelligence research.', 'AI predicts e-commerce engagement.', 'AI in finance streamlines human-computer interaction.', 'AI streamlines human-computer interaction.', 'AI in the legal field is reshaping customer experience.', 'Ethical AI drives advancements in logistics efficiency.', 'Explainable AI is transforming smart city planning.', 'Machine learning is revolutionizing smart city planning.', 'AI-powered virtual assistants is improving supply chain optimization.', 'AI in climate science is improving manufacturing processes.', 'Explainable AI is revolutionizing energy efficiency.', 'AI in education is reshaping financial risk management.', 'Machine learning creates opportunities for human-computer interaction.', 'Edge AI creates opportunities for manufacturing processes.', 'AI in space exploration is reshaping supply chain optimization.', 'AI in healthcare drives advancements in smart city planning.', 'AI advances fraud detection.', 'AI-powered virtual assistants integrates into energy efficiency.', 'AI-powered virtual assistants is reshaping climate science predictions.', 'Big data analytics integrates into artificial intelligence research.', 'AI in healthcare is reshaping smart city planning.', 'AI in marketing accelerates fraud detection.', 'Generative AI drives advancements in document review.', 'AI in cybersecurity is reshaping translation accuracy.', 'AI-powered fraud detection is revolutionizing logistics efficiency.', 'AI in marketing is revolutionizing crop yields.', 'AI in climate science is reshaping fraud detection.', 'AI in climate science is transforming space exploration.', 'AI enhances fraud detection.', 'AI in sports drives advancements in healthcare outcomes.', 'AI in healthcare creates opportunities for human-computer interaction.', 'Explainable AI is revolutionizing cybersecurity systems.', 'AI in marketing reduces costs in human-computer interaction.', 'Ethical AI integrates into manufacturing processes.', 'AI in climate science creates opportunities for supply chain optimization.', 'Machine learning is reshaping fraud detection.', 'Ethical AI is revolutionizing protein structure decoding.', 'AI in healthcare ensures manufacturing processes.', 'Self-driving cars is reshaping financial risk management.', 'AI in sports accelerates supply chain optimization.', 'AI in finance drives advancements in education personalization.', 'AI in marketing aids in energy efficiency.', 'AI in space exploration integrates into human-computer interaction.', 'AI-powered fraud detection is revolutionizing smart city planning.', 'AI in sports is transforming real estate valuation.', 'AI-powered fraud detection is revolutionizing traffic management.', 'AI in the legal field advances e-commerce engagement.', 'AI in sports is reshaping protein structure decoding.', 'AI in logistics is enabling supply chain optimization.', 'Machine learning integrates into manufacturing processes.', 'Explainable AI is reshaping energy efficiency.', 'AI-powered virtual assistants predicts artificial intelligence research.', 'AI-powered fraud detection is improving human-computer interaction.', 'AI in healthcare integrates into supply chain optimization.', 'Smart cities advances artificial intelligence research.', 'Self-driving cars is revolutionizing manufacturing processes.', 'AI in logistics is reshaping traffic management.', 'AI in logistics is revolutionizing customer experience.', 'AI in marketing creates opportunities for financial risk management.', 'AI in marketing advances translation accuracy.', 'AI in finance enhances human-computer interaction.', 'AI in healthcare drives advancements in drug discovery.', 'AI in finance streamlines artificial intelligence research.', 'AI-powered virtual assistants is enabling e-commerce engagement.', 'AI in space exploration is revolutionizing fraud detection.', 'AI in logistics is revolutionizing drug discovery.', 'AI in cybersecurity drives advancements in smart city planning.', 'AI in marketing creates opportunities for e-commerce engagement.', 'Explainable AI drives advancements in space exploration.', 'AI in finance is improving protein structure decoding.', 'AI in healthcare creates opportunities for smart city planning.', 'AI in agriculture is reshaping translation accuracy.', 'AI-powered virtual assistants creates opportunities for energy efficiency.', 'AI in space exploration accelerates manufacturing processes.', 'AI in agriculture is reshaping space exploration.', 'AI improves artificial intelligence research.', 'AI in finance enhances logistics efficiency.', 'AI in marketing aids in e-commerce engagement.', 'AI is revolutionizing protein structure decoding.', 'AI-powered fraud detection creates opportunities for fraud detection.', 'AI in cybersecurity accelerates healthcare outcomes.', 'AI in finance improves fraud detection.', 'AI is revolutionizing climate science predictions.', 'Edge AI is transforming document review.', 'Voice recognition is transforming human-computer interaction.', 'Ethical AI creates opportunities for supply chain optimization.', 'AI in sports drives advancements in space exploration.', 'Machine learning creates opportunities for logistics efficiency.', 'Big data analytics is revolutionizing e-commerce engagement.', 'Ethical AI creates opportunities for energy efficiency.', 'AI in finance integrates into customer experience.', 'AI in marketing is optimizing logistics efficiency.', 'AI in finance predicts fraud detection.', 'AI in cybersecurity is improving translation accuracy.', 'AI in healthcare accelerates drug discovery.', 'AI in cybersecurity simplifies manufacturing processes.', 'Recommendation engines is transforming fraud detection.', 'Edge AI is revolutionizing education personalization.', 'AI in cybersecurity simplifies manufacturing processes.', 'AI in logistics is enabling healthcare outcomes.', 'Neural networks is transforming space exploration.', 'Voice recognition integrates into artificial intelligence research.', 'AI in climate science is revolutionizing crop yields.', 'Machine learning drives advancements in education personalization.', 'Explainable AI creates opportunities for healthcare outcomes.', 'AI in marketing personalizes customer experience.', 'Machine learning aids in manufacturing processes.', 'AI in healthcare drives advancements in education personalization.', 'AI in marketing enhances logistics efficiency.', 'AI-powered fraud detection is enabling healthcare outcomes.', 'AI-powered fraud detection predicts human-computer interaction.', 'AI is improving education personalization.', 'AI in healthcare is improving translation accuracy.', 'AI in agriculture advances document review.', 'AI in logistics is enabling translation accuracy.', 'AI in sports is reshaping education personalization.', 'Ethical AI creates opportunities for fraud detection.', 'AI-powered virtual assistants predicts financial risk management.', 'Recommendation engines predicts artificial intelligence research.', 'AI in finance streamlines supply chain optimization.', 'AI in space exploration enhances artificial intelligence research.', 'AI-powered fraud detection integrates into customer experience.', 'AI in healthcare creates opportunities for protein structure decoding.', 'AI in healthcare streamlines fraud detection.', 'AI in healthcare creates opportunities for energy efficiency.', 'AI in healthcare advances fraud detection.', 'AI in the legal field is revolutionizing energy efficiency.', 'AI in finance is enabling education personalization.', 'AI accelerates real estate valuation.', 'Explainable AI drives advancements in protein structure decoding.', 'AI in marketing drives advancements in traffic management.', 'AI in space exploration enhances artificial intelligence research.', 'Recommendation engines streamlines artificial intelligence research.', 'AI in logistics is optimizing human-computer interaction.', 'Edge AI advances e-commerce engagement.', 'AI in marketing personalizes education personalization.', 'AI integrates into smart city planning.', 'Neural networks drives advancements in player performance in sports.', 'AI in sports drives advancements in education personalization.', 'Edge AI enhances artificial intelligence research.', 'AI in healthcare integrates into financial risk management.', 'AI-powered fraud detection advances education personalization.', 'AI in sports streamlines manufacturing processes.', 'Generative AI aids in manufacturing processes.', 'Neural networks advances e-commerce engagement.', 'AI advances player performance in sports.', 'Smart cities drives advancements in manufacturing processes.', 'Robotics predicts manufacturing processes.', 'Machine learning is optimizing artificial intelligence research.', 'AI in climate science is enabling financial risk management.', 'AI in education drives advancements in healthcare outcomes.', 'AI in agriculture enhances logistics efficiency.', 'AI in logistics streamlines healthcare outcomes.', 'AI-powered fraud detection is enabling drug discovery.', 'AI-powered fraud detection integrates into fraud detection.', 'AI in healthcare is improving logistics efficiency.', 'AI improves smart city planning.', 'AI in sports improves protein structure decoding.', 'AI in the legal field advances supply chain optimization.', 'AI in climate science integrates into human-computer interaction.', 'Generative AI is revolutionizing education personalization.', 'Machine learning aids in e-commerce engagement.', 'Generative AI streamlines financial risk management.', 'Explainable AI accelerates healthcare outcomes.', 'Edge AI personalizes e-commerce engagement.', 'AI-powered virtual assistants aids in crop yields.', 'AI-powered virtual assistants is revolutionizing astronomical data analysis.', 'AI-powered virtual assistants integrates into drug discovery.', 'AI in healthcare creates opportunities for translation accuracy.', 'AI-powered virtual assistants streamlines smart city planning.', 'AI in marketing aids in artificial intelligence research.', 'AI in healthcare is enabling space exploration.', 'AI in finance drives advancements in astronomical data analysis.', 'AI in the legal field improves artificial intelligence research.', 'AI in sports accelerates protein structure decoding.', 'AI in marketing predicts climate science predictions.', 'AI in finance creates opportunities for protein structure decoding.', 'AI in finance drives advancements in astronomical data analysis.', 'AI in finance is enabling space exploration.', 'AI-powered virtual assistants is optimizing financial risk management.', 'AI in healthcare aids in e-commerce engagement.', 'AI in the legal field advances drug discovery.', 'AI in education drives advancements in traffic management.', 'AI in climate science drives advancements in translation accuracy.', 'AI in logistics advances customer experience.', 'AI in climate science ensures e-commerce engagement.', 'AI-powered fraud detection advances logistics efficiency.', 'Recommendation engines is revolutionizing real estate valuation.', 'AI in the legal field is enabling fraud detection.', 'Machine learning is improving fraud detection.', 'AI in finance drives advancements in climate science predictions.', 'Self-driving cars is improving manufacturing processes.', 'AI in logistics aids in protein structure decoding.', 'AI in logistics drives advancements in climate science predictions.', 'AI in logistics enhances e-commerce engagement.', 'AI in finance is optimizing financial risk management.', 'AI in marketing is optimizing supply chain optimization.', 'AI in climate science improves smart city planning.', 'Edge AI is revolutionizing drug discovery.', 'AI in logistics is improving customer experience.', 'Robotics is reshaping e-commerce engagement.', 'AI in climate science is revolutionizing astronomical data analysis.', 'Quantum computing drives advancements in fraud detection.', 'Edge AI is revolutionizing healthcare outcomes.', 'AI in the legal field simplifies manufacturing processes.', 'AI in marketing is improving space exploration.', 'Transfer learning is revolutionizing human-computer interaction.', 'AI in education is reshaping climate science predictions.', 'Ethical AI advances financial risk management.', 'AI in marketing creates opportunities for climate science predictions.', 'AI in agriculture simplifies fraud detection.', 'AI in logistics is improving document review.', 'Quantum computing is transforming fraud detection.', 'Natural language processing is revolutionizing real estate valuation.', 'AI in cybersecurity advances financial risk management.', 'AI-powered virtual assistants simplifies education personalization.', 'AI in space exploration creates opportunities for fraud detection.', 'Natural language processing is enabling manufacturing processes.', 'AI in logistics simplifies healthcare outcomes.', 'AI in finance reduces costs in e-commerce engagement.', 'AI in climate science integrates into financial risk management.', 'AI-powered virtual assistants simplifies education personalization.', 'Machine learning personalizes financial risk management.', 'AI in sports is improving document review.', 'AI in agriculture predicts drug discovery.', 'AI accelerates astronomical data analysis.', 'AI in agriculture is optimizing supply chain optimization.', 'Robotics is transforming financial risk management.', 'AI in agriculture integrates into artificial intelligence research.', 'Generative AI integrates into artificial intelligence research.', 'AI simplifies supply chain optimization.', 'AI in marketing accelerates player performance in sports.', 'AI in healthcare aids in financial risk management.', 'AI-powered fraud detection creates opportunities for protein structure decoding.', 'AI-powered virtual assistants creates opportunities for real estate valuation.', 'AI creates opportunities for education personalization.', 'AI in climate science is optimizing customer experience.', 'AI in cybersecurity is optimizing protein structure decoding.', 'AI in cybersecurity integrates into financial risk management.', 'Robotics is transforming document review.', 'Explainable AI accelerates climate science predictions.', 'AI in education is enabling protein structure decoding.', 'AI in marketing is optimizing real estate valuation.', 'AI in finance aids in logistics efficiency.', 'Generative AI integrates into fraud detection.', 'AI in agriculture improves document review.', 'Big data analytics drives advancements in financial risk management.', 'AI in marketing creates opportunities for player performance in sports.', 'AI in cybersecurity simplifies supply chain optimization.', 'AI integrates into traffic management.', 'AI in climate science is improving fraud detection.', 'AI in healthcare enhances education personalization.', 'AI in marketing integrates into document review.', 'AI integrates into cybersecurity systems.', 'AI-powered fraud detection is optimizing customer experience.', 'Voice recognition is improving artificial intelligence research.', 'AI-powered fraud detection is enabling smart city planning.', 'AI-powered virtual assistants is enabling energy efficiency.', 'AI in sports streamlines protein structure decoding.', 'AI-powered fraud detection integrates into cybersecurity systems.', 'AI in education is improving energy efficiency.', 'Explainable AI accelerates player performance in sports.', 'AI-powered fraud detection is optimizing e-commerce engagement.', 'Self-driving cars drives advancements in translation accuracy.', 'AI in cybersecurity improves energy efficiency.', 'AI in finance advances healthcare outcomes.', 'AI in marketing predicts traffic management.', 'AI in marketing aids in smart city planning.', 'Machine learning is optimizing logistics efficiency.', 'AI in logistics is optimizing protein structure decoding.', 'Edge AI drives advancements in logistics efficiency.', 'AI-powered fraud detection advances protein structure decoding.', 'AI in space exploration is improving customer experience.', 'Machine learning enhances human-computer interaction.', 'AI in climate science improves supply chain optimization.', 'AI-powered fraud detection streamlines supply chain optimization.', 'Explainable AI is improving astronomical data analysis.', 'AI in logistics streamlines translation accuracy.', 'AI in the legal field is improving crop yields.', 'AI in education is enabling translation accuracy.', 'AI in healthcare accelerates space exploration.', 'AI in marketing is optimizing space exploration.', 'AI in agriculture integrates into real estate valuation.', 'AI in marketing is improving real estate valuation.', 'AI in marketing is improving real estate valuation.', 'AI in marketing personalizes climate science predictions.', 'AI-powered virtual assistants aids in education personalization.', 'Explainable AI streamlines crop yields.', 'AI in agriculture advances real estate valuation.', 'AI in the legal field predicts artificial intelligence research.', 'Robotics is reshaping financial risk management.', 'Transfer learning is transforming player performance in sports.', 'AI in finance reduces costs in human-computer interaction.', 'AI in finance accelerates document review.', 'AI in the legal field advances protein structure decoding.', 'AI in marketing is optimizing document review.', 'Voice recognition is transforming logistics efficiency.', 'AI in education integrates into financial risk management.', 'Smart cities is transforming logistics efficiency.', 'AI-powered virtual assistants personalizes healthcare outcomes.', 'Ethical AI enhances healthcare outcomes.', 'AI in marketing simplifies smart city planning.', 'AI in cybersecurity accelerates space exploration.', 'AI-powered virtual assistants improves climate science predictions.', 'AI in space exploration aids in healthcare outcomes.', 'AI in finance accelerates climate science predictions.', 'AI in logistics improves astronomical data analysis.', 'AI-powered virtual assistants is improving translation accuracy.', 'AI improves financial risk management.', 'AI in cybersecurity enhances real estate valuation.', 'Generative AI is improving protein structure decoding.', 'AI in cybersecurity personalizes customer experience.', 'AI in logistics improves customer experience.', 'AI in logistics advances healthcare outcomes.', 'AI in marketing accelerates document review.', 'AI in sports improves healthcare outcomes.', 'AI in sports is enabling cybersecurity systems.', 'AI in healthcare creates opportunities for space exploration.', 'AI in sports integrates into translation accuracy.', 'AI-powered fraud detection improves protein structure decoding.', 'AI-powered virtual assistants personalizes astronomical data analysis.', 'AI in climate science ensures energy efficiency.', 'AI in logistics advances space exploration.', 'AI-powered virtual assistants integrates into translation accuracy.', 'AI in space exploration is reshaping crop yields.', 'AI in agriculture streamlines climate science predictions.', 'AI in cybersecurity enhances smart city planning.', 'Natural language processing is reshaping document review.', 'AI in climate science accelerates logistics efficiency.', 'Generative AI advances drug discovery.', 'AI in sports personalizes manufacturing processes.', 'AI aids in space exploration.', 'AI in the legal field is enabling crop yields.', 'AI in education simplifies fraud detection.', 'Voice recognition drives advancements in human-computer interaction.', 'AI in climate science enhances logistics efficiency.', 'AI in climate science predicts artificial intelligence research.', 'Reinforcement learning is improving manufacturing processes.', 'AI in logistics enhances customer experience.', 'AI in healthcare reduces costs in translation accuracy.', 'Explainable AI is improving energy efficiency.', 'Machine learning accelerates energy efficiency.', 'AI in agriculture accelerates document review.', 'Explainable AI is enabling crop yields.', 'AI in cybersecurity simplifies human-computer interaction.', 'AI in sports is optimizing education personalization.', 'Machine learning advances real estate valuation.', 'AI in cybersecurity streamlines translation accuracy.', 'Machine learning enhances healthcare outcomes.', 'AI in space exploration creates opportunities for healthcare outcomes.', 'AI in education ensures protein structure decoding.', 'AI in education ensures real estate valuation.', 'Edge AI is enabling human-computer interaction.', 'Machine learning drives advancements in translation accuracy.', 'AI in healthcare reduces costs in document review.', 'AI-powered fraud detection predicts protein structure decoding.', 'AI in space exploration personalizes logistics efficiency.', 'Explainable AI is enabling crop yields.', 'AI in logistics predicts customer experience.', 'AI in the legal field improves crop yields.', 'AI-powered fraud detection improves translation accuracy.', 'Neural networks drives advancements in astronomical data analysis.', 'AI-powered fraud detection improves smart city planning.', 'AI in agriculture predicts healthcare outcomes.', 'AI in climate science streamlines traffic management.', 'AI simplifies energy efficiency.', 'Recommendation engines drives advancements in real estate valuation.', 'Generative AI advances supply chain optimization.', 'Explainable AI is improving customer experience.', 'AI-powered fraud detection predicts logistics efficiency.', 'AI in logistics accelerates player performance in sports.', 'Machine learning ensures e-commerce engagement.', 'AI in healthcare accelerates traffic management.', 'Transfer learning drives advancements in human-computer interaction.', 'AI in logistics predicts financial risk management.', 'AI in education creates opportunities for astronomical data analysis.', 'Natural language processing is revolutionizing protein structure decoding.', 'AI in sports advances cybersecurity systems.', 'Self-driving cars predicts e-commerce engagement.', 'AI in space exploration creates opportunities for healthcare outcomes.', 'AI in healthcare advances cybersecurity systems.', 'AI in sports streamlines energy efficiency.', 'Neural networks enhances fraud detection.', 'AI-powered fraud detection is improving climate science predictions.', 'Explainable AI enhances space exploration.', 'AI in agriculture integrates into customer experience.', 'Self-driving cars integrates into e-commerce engagement.', 'AI in healthcare predicts drug discovery.', 'AI in agriculture is optimizing energy efficiency.', 'AI-powered fraud detection is enabling document review.', 'AI in cybersecurity improves translation accuracy.', 'AI in finance is improving traffic management.', 'Smart cities simplifies artificial intelligence research.', 'Natural language processing is transforming astronomical data analysis.', 'AI in sports creates opportunities for climate science predictions.', 'AI-powered fraud detection creates opportunities for space exploration.', 'AI in finance enhances climate science predictions.', 'AI in cybersecurity is enabling space exploration.', 'Neural networks integrates into drug discovery.', 'AI in logistics is optimizing cybersecurity systems.', 'Neural networks is enabling translation accuracy.', 'Ethical AI predicts energy efficiency.', 'AI in space exploration reduces costs in drug discovery.', 'AI in cybersecurity aids in translation accuracy.', 'Ethical AI predicts financial risk management.', 'AI streamlines player performance in sports.', 'AI in finance improves translation accuracy.', 'Reinforcement learning is revolutionizing smart city planning.', 'Robotics is transforming education personalization.', 'AI in cybersecurity accelerates fraud detection.', 'AI in agriculture reduces costs in artificial intelligence research.', 'Neural networks enhances crop yields.', 'AI in space exploration aids in crop yields.', 'Neural networks integrates into energy efficiency.', 'AI in climate science aids in financial risk management.', 'Big data analytics is reshaping e-commerce engagement.', 'Generative AI is improving customer experience.', 'AI in agriculture simplifies drug discovery.', 'Machine learning predicts drug discovery.', 'AI in logistics enhances astronomical data analysis.', 'AI in healthcare is optimizing financial risk management.', 'AI in education enhances crop yields.', 'Generative AI is enabling drug discovery.', 'AI in cybersecurity is optimizing real estate valuation.', 'AI-powered fraud detection ensures logistics efficiency.', 'AI in climate science aids in financial risk management.', 'AI in space exploration personalizes e-commerce engagement.', 'Explainable AI is enabling climate science predictions.', 'AI in cybersecurity enhances player performance in sports.', 'AI in finance is optimizing climate science predictions.', 'AI in logistics is optimizing traffic management.']}
2025-02-14 13:04:55,679 [INFO] Rerank API response: {'id': '1bbe9b40-f065-458e-b2fb-82d2be43ac3e', 'results': [{'index': 1, 'relevance_score': 0.5393963}, {'index': 10, 'relevance_score': 0.5321097}, {'index': 19, 'relevance_score': 0.5253938}, {'index': 15, 'relevance_score': 0.52100873}, {'index': 18, 'relevance_score': 0.5090084}, {'index': 16, 'relevance_score': 0.50183165}, {'index': 31, 'relevance_score': 0.48996773}, {'index': 6, 'relevance_score': 0.48806435}, {'index': 3, 'relevance_score': 0.48484412}, {'index': 21, 'relevance_score': 0.4813325}, {'index': 0, 'relevance_score': 0.4757764}, {'index': 8, 'relevance_score': 0.47431523}, {'index': 28, 'relevance_score': 0.47314653}, {'index': 113, 'relevance_score': 0.46687052}, {'index': 96, 'relevance_score': 0.46016815}, {'index': 9, 'relevance_score': 0.45159313}, {'index': 75, 'relevance_score': 0.45159313}, {'index': 22, 'relevance_score': 0.45057756}, {'index': 54, 'relevance_score': 0.4430467}, {'index': 11, 'relevance_score': 0.44261298}, {'index': 38, 'relevance_score': 0.43799233}, {'index': 61, 'relevance_score': 0.43698296}, {'index': 27, 'relevance_score': 0.43352634}, {'index': 2, 'relevance_score': 0.43338245}, {'index': 39, 'relevance_score': 0.43223172}, {'index': 4, 'relevance_score': 0.42105237}, {'index': 152, 'relevance_score': 0.42062396}, {'index': 49, 'relevance_score': 0.41762826}, {'index': 29, 'relevance_score': 0.41534993}, {'index': 30, 'relevance_score': 0.41392776}, {'index': 7, 'relevance_score': 0.4120811}, {'index': 32, 'relevance_score': 0.39543822}, {'index': 44, 'relevance_score': 0.39208132}, {'index': 40, 'relevance_score': 0.3869261}, {'index': 12, 'relevance_score': 0.38477397}, {'index': 25, 'relevance_score': 0.3787583}, {'index': 24, 'relevance_score': 0.37401357}, {'index': 36, 'relevance_score': 0.37175277}, {'index': 20, 'relevance_score': 0.3694976}, {'index': 17, 'relevance_score': 0.36772484}, {'index': 55, 'relevance_score': 0.36107516}, {'index': 176, 'relevance_score': 0.35716435}, {'index': 126, 'relevance_score': 0.35033303}, {'index': 89, 'relevance_score': 0.34323218}, {'index': 391, 'relevance_score': 0.34283602}, {'index': 449, 'relevance_score': 0.34237418}, {'index': 357, 'relevance_score': 0.33137986}, {'index': 143, 'relevance_score': 0.3292089}, {'index': 62, 'relevance_score': 0.32727093}, {'index': 78, 'relevance_score': 0.32566044}], 'meta': {'api_version': {'version': '2'}, 'billed_units': {'search_units': 5}}}
2025-02-14 13:04:55,680 [INFO] Time to retrieve reranked results: 1.25 seconds
2025-02-14 13:04:55,680 [INFO] Reranked Results:
2025-02-14 13:04:55,680 [INFO]  DocID=851, Score=0.5394, Doc='AI is transforming supply chain optimization.'
2025-02-14 13:04:55,680 [INFO]  DocID=659, Score=0.5321, Doc='AI in sports is transforming manufacturing processes.'
2025-02-14 13:04:55,680 [INFO]  DocID=827, Score=0.5254, Doc='Explainable AI is transforming e-commerce engagement.'
2025-02-14 13:04:55,680 [INFO]  DocID=871, Score=0.5210, Doc='Smart cities is transforming artificial intelligence research.'
2025-02-14 13:04:55,680 [INFO]  DocID=764, Score=0.5090, Doc='AI in agriculture is transforming education personalization.'
2025-02-14 13:04:55,680 [INFO]  DocID=590, Score=0.5018, Doc='AI in logistics is transforming smart city planning.'
2025-02-14 13:04:55,680 [INFO]  DocID=839, Score=0.4900, Doc='AI in logistics is transforming education personalization.'
2025-02-14 13:04:55,680 [INFO]  DocID=430, Score=0.4881, Doc='AI in marketing is transforming e-commerce engagement.'
2025-02-14 13:04:55,680 [INFO]  DocID=124, Score=0.4848, Doc='AI in logistics is transforming fraud detection.'
2025-02-14 13:04:55,680 [INFO]  DocID=242, Score=0.4813, Doc='AI is transforming cybersecurity systems.'
2025-02-14 13:04:55,680 [INFO]  DocID=620, Score=0.4758, Doc='AI in marketing is transforming fraud detection.'
2025-02-14 13:04:55,681 [INFO]  DocID=510, Score=0.4743, Doc='AI is transforming smart city planning.'
2025-02-14 13:04:55,681 [INFO]  DocID=377, Score=0.4731, Doc='AI in cybersecurity is transforming real estate valuation.'
2025-02-14 13:04:55,681 [INFO]  DocID=717, Score=0.4669, Doc='AI in sports is transforming real estate valuation.'
2025-02-14 13:04:55,681 [INFO]  DocID=237, Score=0.4602, Doc='AI in climate science is transforming space exploration.'
2025-02-14 13:04:55,681 [INFO]  DocID=434, Score=0.4516, Doc='AI in marketing is transforming player performance in sports.'
2025-02-14 13:04:55,681 [INFO]  DocID=525, Score=0.4516, Doc='Explainable AI is transforming smart city planning.'
2025-02-14 13:04:55,681 [INFO]  DocID=260, Score=0.4506, Doc='AI in healthcare is revolutionizing manufacturing processes.'
2025-02-14 13:04:55,681 [INFO]  DocID=897, Score=0.4430, Doc='AI in climate science is transforming education personalization.'
2025-02-14 13:04:55,681 [INFO]  DocID=282, Score=0.4426, Doc='AI in climate science is transforming financial risk management.'
2025-02-14 13:04:55,681 [INFO]  DocID=653, Score=0.4380, Doc='AI-powered virtual assistants is transforming e-commerce engagement.'
2025-02-14 13:04:55,681 [INFO]  DocID=327, Score=0.4370, Doc='AI is transforming traffic management.'
2025-02-14 13:04:55,681 [INFO]  DocID=52, Score=0.4335, Doc='AI in marketing is transforming astronomical data analysis.'
2025-02-14 13:04:55,681 [INFO]  DocID=695, Score=0.4334, Doc='AI in education is revolutionizing manufacturing processes.'
2025-02-14 13:04:55,681 [INFO]  DocID=47, Score=0.4322, Doc='AI-powered fraud detection is transforming e-commerce engagement.'
2025-02-14 13:04:55,681 [INFO]  DocID=485, Score=0.4211, Doc='Quantum computing is transforming artificial intelligence research.'
2025-02-14 13:04:55,681 [INFO]  DocID=229, Score=0.4206, Doc='Edge AI is transforming document review.'
2025-02-14 13:04:55,681 [INFO]  DocID=28, Score=0.4176, Doc='AI in the legal field is transforming crop yields.'
2025-02-14 13:04:55,681 [INFO]  DocID=581, Score=0.4153, Doc='AI in finance is transforming crop yields.'
2025-02-14 13:04:55,681 [INFO]  DocID=177, Score=0.4139, Doc='AI in marketing is transforming translation accuracy.'
2025-02-14 13:04:55,681 [INFO]  DocID=952, Score=0.4121, Doc='AI in cybersecurity is revolutionizing manufacturing processes.'
2025-02-14 13:04:55,681 [INFO]  DocID=219, Score=0.3954, Doc='Machine learning is transforming supply chain optimization.'
2025-02-14 13:04:55,681 [INFO]  DocID=309, Score=0.3921, Doc='AI in space exploration is transforming customer experience.'
2025-02-14 13:04:55,681 [INFO]  DocID=240, Score=0.3869, Doc='AI is transforming player performance in sports.'
2025-02-14 13:04:55,681 [INFO]  DocID=548, Score=0.3848, Doc='AI is revolutionizing human-computer interaction.'
2025-02-14 13:04:55,681 [INFO]  DocID=643, Score=0.3788, Doc='Ethical AI is revolutionizing manufacturing processes.'
2025-02-14 13:04:55,681 [INFO]  DocID=375, Score=0.3740, Doc='AI in logistics is improving manufacturing processes.'
2025-02-14 13:04:55,681 [INFO]  DocID=734, Score=0.3718, Doc='AI in finance is revolutionizing healthcare outcomes.'
2025-02-14 13:04:55,681 [INFO]  DocID=25, Score=0.3695, Doc='AI in marketing is improving manufacturing processes.'
2025-02-14 13:04:55,681 [INFO]  DocID=597, Score=0.3677, Doc='AI in healthcare is revolutionizing supply chain optimization.'
2025-02-14 13:04:55,681 [INFO]  DocID=503, Score=0.3611, Doc='AI in healthcare is revolutionizing logistics efficiency.'
2025-02-14 13:04:55,681 [INFO]  DocID=673, Score=0.3572, Doc='AI in healthcare drives advancements in education personalization.'
2025-02-14 13:04:55,681 [INFO]  DocID=670, Score=0.3503, Doc='AI in logistics is revolutionizing customer experience.'
2025-02-14 13:04:55,681 [INFO]  DocID=329, Score=0.3432, Doc='AI in healthcare is reshaping smart city planning.'
2025-02-14 13:04:55,682 [INFO]  DocID=750, Score=0.3428, Doc='AI in logistics advances space exploration.'
2025-02-14 13:04:55,682 [INFO]  DocID=644, Score=0.3424, Doc='AI in healthcare advances cybersecurity systems.'
2025-02-14 13:04:55,682 [INFO]  DocID=865, Score=0.3314, Doc='AI in agriculture advances real estate valuation.'
2025-02-14 13:04:55,682 [INFO]  DocID=718, Score=0.3292, Doc='AI in agriculture is reshaping space exploration.'
2025-02-14 13:04:55,682 [INFO]  DocID=487, Score=0.3273, Doc='AI in finance is revolutionizing climate science predictions.'
2025-02-14 13:04:55,682 [INFO]  DocID=145, Score=0.3257, Doc='AI in climate science is improving manufacturing processes.'
2025-02-14 13:04:55,683 [INFO] 
=== Compare Top-10 Float vs. Int8 ===
2025-02-14 13:04:55,686 [INFO] Existing float FAISS index loaded.
2025-02-14 13:04:56,187 [INFO]  --- Top-10 Float ---
2025-02-14 13:04:56,187 [INFO]   DocID=851, Score=0.5896, Doc='AI is transforming supply chain optimization.'
2025-02-14 13:04:56,187 [INFO]   DocID=659, Score=0.5761, Doc='AI in sports is transforming manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=952, Score=0.5742, Doc='AI in cybersecurity is revolutionizing manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=548, Score=0.5650, Doc='AI is revolutionizing human-computer interaction.'
2025-02-14 13:04:56,188 [INFO]   DocID=242, Score=0.5616, Doc='AI is transforming cybersecurity systems.'
2025-02-14 13:04:56,188 [INFO]   DocID=620, Score=0.5532, Doc='AI in marketing is transforming fraud detection.'
2025-02-14 13:04:56,188 [INFO]   DocID=25, Score=0.5519, Doc='AI in marketing is improving manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=260, Score=0.5518, Doc='AI in healthcare is revolutionizing manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=643, Score=0.5489, Doc='Ethical AI is revolutionizing manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=510, Score=0.5468, Doc='AI is transforming smart city planning.'
2025-02-14 13:04:56,188 [INFO]   DocID=430, Score=0.5447, Doc='AI in marketing is transforming e-commerce engagement.'
2025-02-14 13:04:56,188 [INFO]   DocID=327, Score=0.5419, Doc='AI is transforming traffic management.'
2025-02-14 13:04:56,188 [INFO]   DocID=695, Score=0.5411, Doc='AI in education is revolutionizing manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=590, Score=0.5375, Doc='AI in logistics is transforming smart city planning.'
2025-02-14 13:04:56,188 [INFO]   DocID=703, Score=0.5356, Doc='AI-powered virtual assistants creates opportunities for manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=653, Score=0.5328, Doc='AI-powered virtual assistants is transforming e-commerce engagement.'
2025-02-14 13:04:56,188 [INFO]   DocID=124, Score=0.5328, Doc='AI in logistics is transforming fraud detection.'
2025-02-14 13:04:56,188 [INFO]   DocID=375, Score=0.5326, Doc='AI in logistics is improving manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=463, Score=0.5292, Doc='AI in marketing enhances manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=177, Score=0.5285, Doc='AI in marketing is transforming translation accuracy.'
2025-02-14 13:04:56,188 [INFO]   DocID=827, Score=0.5273, Doc='Explainable AI is transforming e-commerce engagement.'
2025-02-14 13:04:56,188 [INFO]   DocID=485, Score=0.5231, Doc='Quantum computing is transforming artificial intelligence research.'
2025-02-14 13:04:56,188 [INFO]   DocID=597, Score=0.5224, Doc='AI in healthcare is revolutionizing supply chain optimization.'
2025-02-14 13:04:56,188 [INFO]   DocID=240, Score=0.5209, Doc='AI is transforming player performance in sports.'
2025-02-14 13:04:56,188 [INFO]   DocID=864, Score=0.5196, Doc='Self-driving cars drives advancements in artificial intelligence research.'
2025-02-14 13:04:56,188 [INFO]   DocID=52, Score=0.5192, Doc='AI in marketing is transforming astronomical data analysis.'
2025-02-14 13:04:56,188 [INFO]   DocID=434, Score=0.5178, Doc='AI in marketing is transforming player performance in sports.'
2025-02-14 13:04:56,188 [INFO]   DocID=871, Score=0.5176, Doc='Smart cities is transforming artificial intelligence research.'
2025-02-14 13:04:56,188 [INFO]   DocID=47, Score=0.5152, Doc='AI-powered fraud detection is transforming e-commerce engagement.'
2025-02-14 13:04:56,188 [INFO]   DocID=874, Score=0.5144, Doc='AI-powered fraud detection is improving manufacturing processes.'
2025-02-14 13:04:56,188 [INFO]   DocID=499, Score=0.5142, Doc='AI streamlines human-computer interaction.'
2025-02-14 13:04:56,188 [INFO]   DocID=525, Score=0.5108, Doc='Explainable AI is transforming smart city planning.'
2025-02-14 13:04:56,189 [INFO]   DocID=839, Score=0.5096, Doc='AI in logistics is transforming education personalization.'
2025-02-14 13:04:56,189 [INFO]   DocID=381, Score=0.5095, Doc='AI-powered virtual assistants personalizes manufacturing processes.'
2025-02-14 13:04:56,189 [INFO]   DocID=440, Score=0.5076, Doc='AI in sports is revolutionizing supply chain optimization.'
2025-02-14 13:04:56,189 [INFO]   DocID=627, Score=0.5056, Doc='Generative AI is revolutionizing artificial intelligence research.'
2025-02-14 13:04:56,189 [INFO]   DocID=670, Score=0.5046, Doc='AI in logistics is revolutionizing customer experience.'
2025-02-14 13:04:56,189 [INFO]   DocID=208, Score=0.5043, Doc='AI drives advancements in fraud detection.'
2025-02-14 13:04:56,189 [INFO]   DocID=521, Score=0.5039, Doc='AI-powered fraud detection is revolutionizing logistics efficiency.'
2025-02-14 13:04:56,189 [INFO]   DocID=503, Score=0.5026, Doc='AI in healthcare is revolutionizing logistics efficiency.'
2025-02-14 13:04:56,189 [INFO]   DocID=219, Score=0.5012, Doc='Machine learning is transforming supply chain optimization.'
2025-02-14 13:04:56,189 [INFO]   DocID=675, Score=0.5008, Doc='AI-powered virtual assistants drives advancements in customer experience.'
2025-02-14 13:04:56,189 [INFO]   DocID=832, Score=0.4977, Doc='AI is revolutionizing document review.'
2025-02-14 13:04:56,189 [INFO]   DocID=858, Score=0.4967, Doc='AI is improving e-commerce engagement.'
2025-02-14 13:04:56,189 [INFO]   DocID=282, Score=0.4961, Doc='AI in climate science is transforming financial risk management.'
2025-02-14 13:04:56,189 [INFO]   DocID=791, Score=0.4961, Doc='Explainable AI is revolutionizing e-commerce engagement.'
2025-02-14 13:04:56,189 [INFO]   DocID=110, Score=0.4950, Doc='AI in marketing is revolutionizing crop yields.'
2025-02-14 13:04:56,189 [INFO]   DocID=822, Score=0.4944, Doc='AI-powered fraud detection is reshaping customer experience.'
2025-02-14 13:04:56,189 [INFO]   DocID=145, Score=0.4940, Doc='AI in climate science is improving manufacturing processes.'
2025-02-14 13:04:56,189 [INFO]   DocID=581, Score=0.4926, Doc='AI in finance is transforming crop yields.'
2025-02-14 13:04:56,190 [INFO] Existing FAISS binary index loaded.
2025-02-14 13:04:56,693 [INFO] Existing float FAISS index loaded.
2025-02-14 13:04:57,192 [INFO]  --- Top-10 Int8 ---
2025-02-14 13:04:57,192 [INFO]   DocID=28, Score=0.5896, Doc='AI in the legal field is transforming crop yields.'
2025-02-14 13:04:57,192 [INFO]   DocID=381, Score=0.5865, Doc='AI-powered virtual assistants personalizes manufacturing processes.'
2025-02-14 13:04:57,192 [INFO]   DocID=429, Score=0.5865, Doc='AI in finance creates opportunities for e-commerce engagement.'
2025-02-14 13:04:57,192 [INFO]   DocID=197, Score=0.5833, Doc='Explainable AI drives advancements in document review.'
2025-02-14 13:04:57,192 [INFO]   DocID=309, Score=0.5833, Doc='AI in space exploration is transforming customer experience.'
2025-02-14 13:04:57,192 [INFO]   DocID=473, Score=0.5833, Doc='AI in education drives advancements in human-computer interaction.'
2025-02-14 13:04:57,192 [INFO]   DocID=991, Score=0.5833, Doc='AI in healthcare is reshaping energy efficiency.'
2025-02-14 13:04:57,192 [INFO]   DocID=47, Score=0.5802, Doc='AI-powered fraud detection is transforming e-commerce engagement.'
2025-02-14 13:04:57,192 [INFO]   DocID=240, Score=0.5802, Doc='AI is transforming player performance in sports.'
2025-02-14 13:04:57,192 [INFO]   DocID=604, Score=0.5802, Doc='AI in education drives advancements in e-commerce engagement.'
2025-02-14 13:04:57,192 [INFO]   DocID=661, Score=0.5802, Doc='AI in cybersecurity is reshaping financial risk management.'
2025-02-14 13:04:57,193 [INFO]   DocID=653, Score=0.5739, Doc='AI-powered virtual assistants is transforming e-commerce engagement.'
2025-02-14 13:04:57,193 [INFO]   DocID=734, Score=0.5708, Doc='AI in finance is revolutionizing healthcare outcomes.'
2025-02-14 13:04:57,193 [INFO]   DocID=832, Score=0.5708, Doc='AI is revolutionizing document review.'
2025-02-14 13:04:57,193 [INFO]   DocID=219, Score=0.5677, Doc='Machine learning is transforming supply chain optimization.'
2025-02-14 13:04:57,193 [INFO]   DocID=440, Score=0.5677, Doc='AI in sports is revolutionizing supply chain optimization.'
2025-02-14 13:04:57,193 [INFO]   DocID=612, Score=0.5677, Doc='Explainable AI drives advancements in fraud detection.'
2025-02-14 13:04:57,193 [INFO]   DocID=791, Score=0.5677, Doc='Explainable AI is revolutionizing e-commerce engagement.'
2025-02-14 13:04:57,193 [INFO]   DocID=177, Score=0.5646, Doc='AI in marketing is transforming translation accuracy.'
2025-02-14 13:04:57,193 [INFO]   DocID=839, Score=0.5646, Doc='AI in logistics is transforming education personalization.'
2025-02-14 13:04:57,193 [INFO]   DocID=30, Score=0.5614, Doc='AI in the legal field is reshaping artificial intelligence research.'
2025-02-14 13:04:57,193 [INFO]   DocID=52, Score=0.5614, Doc='AI in marketing is transforming astronomical data analysis.'
2025-02-14 13:04:57,193 [INFO]   DocID=377, Score=0.5614, Doc='AI in cybersecurity is transforming real estate valuation.'
2025-02-14 13:04:57,193 [INFO]   DocID=581, Score=0.5614, Doc='AI in finance is transforming crop yields.'
2025-02-14 13:04:57,193 [INFO]   DocID=643, Score=0.5583, Doc='Ethical AI is revolutionizing manufacturing processes.'
2025-02-14 13:04:57,193 [INFO]   DocID=375, Score=0.5552, Doc='AI in logistics is improving manufacturing processes.'
2025-02-14 13:04:57,193 [INFO]   DocID=25, Score=0.5489, Doc='AI in marketing is improving manufacturing processes.'
2025-02-14 13:04:57,193 [INFO]   DocID=242, Score=0.5489, Doc='AI is transforming cybersecurity systems.'
2025-02-14 13:04:57,193 [INFO]   DocID=260, Score=0.5489, Doc='AI in healthcare is revolutionizing manufacturing processes.'
2025-02-14 13:04:57,193 [INFO]   DocID=463, Score=0.5489, Doc='AI in marketing enhances manufacturing processes.'
2025-02-14 13:04:57,193 [INFO]   DocID=590, Score=0.5458, Doc='AI in logistics is transforming smart city planning.'
2025-02-14 13:04:57,193 [INFO]   DocID=597, Score=0.5458, Doc='AI in healthcare is revolutionizing supply chain optimization.'
2025-02-14 13:04:57,193 [INFO]   DocID=764, Score=0.5458, Doc='AI in agriculture is transforming education personalization.'
2025-02-14 13:04:57,193 [INFO]   DocID=827, Score=0.5458, Doc='Explainable AI is transforming e-commerce engagement.'
2025-02-14 13:04:57,193 [INFO]   DocID=208, Score=0.5426, Doc='AI drives advancements in fraud detection.'
2025-02-14 13:04:57,193 [INFO]   DocID=871, Score=0.5426, Doc='Smart cities is transforming artificial intelligence research.'
2025-02-14 13:04:57,193 [INFO]   DocID=282, Score=0.5364, Doc='AI in climate science is transforming financial risk management.'
2025-02-14 13:04:57,193 [INFO]   DocID=548, Score=0.5364, Doc='AI is revolutionizing human-computer interaction.'
2025-02-14 13:04:57,193 [INFO]   DocID=864, Score=0.5364, Doc='Self-driving cars drives advancements in artificial intelligence research.'
2025-02-14 13:04:57,193 [INFO]   DocID=659, Score=0.5301, Doc='AI in sports is transforming manufacturing processes.'
2025-02-14 13:04:57,193 [INFO]   DocID=434, Score=0.5270, Doc='AI in marketing is transforming player performance in sports.'
2025-02-14 13:04:57,193 [INFO]   DocID=510, Score=0.5239, Doc='AI is transforming smart city planning.'
2025-02-14 13:04:57,193 [INFO]   DocID=430, Score=0.5207, Doc='AI in marketing is transforming e-commerce engagement.'
2025-02-14 13:04:57,193 [INFO]   DocID=952, Score=0.5207, Doc='AI in cybersecurity is revolutionizing manufacturing processes.'
2025-02-14 13:04:57,194 [INFO]   DocID=124, Score=0.5176, Doc='AI in logistics is transforming fraud detection.'
2025-02-14 13:04:57,194 [INFO]   DocID=485, Score=0.5176, Doc='Quantum computing is transforming artificial intelligence research.'
2025-02-14 13:04:57,194 [INFO]   DocID=703, Score=0.5176, Doc='AI-powered virtual assistants creates opportunities for manufacturing processes.'
2025-02-14 13:04:57,194 [INFO]   DocID=695, Score=0.5113, Doc='AI in education is revolutionizing manufacturing processes.'
2025-02-14 13:04:57,194 [INFO]   DocID=851, Score=0.4957, Doc='AI is transforming supply chain optimization.'
2025-02-14 13:04:57,194 [INFO]   DocID=620, Score=0.4926, Doc='AI in marketing is transforming fraud detection.'
2025-02-14 13:04:57,194 [INFO] 
=== Float vs. Int8 (Top-10 Overlap) ===
2025-02-14 13:04:57,194 [INFO] DocID=260, Float=0.5518, Int8=0.5489, Diff=0.0029
2025-02-14 13:04:57,194 [INFO] DocID=521 => ONLY in Float top10, Score=0.5039
2025-02-14 13:04:57,194 [INFO] DocID=525 => ONLY in Float top10, Score=0.5108
2025-02-14 13:04:57,194 [INFO] DocID=791, Float=0.4961, Int8=0.5677, Diff=0.0716
2025-02-14 13:04:57,194 [INFO] DocID=25, Float=0.5519, Int8=0.5489, Diff=0.0030
2025-02-14 13:04:57,194 [INFO] DocID=282, Float=0.4961, Int8=0.5364, Diff=0.0403
2025-02-14 13:04:57,194 [INFO] DocID=28 => ONLY in Int8 top10, Score=0.5896
2025-02-14 13:04:57,194 [INFO] DocID=30 => ONLY in Int8 top10, Score=0.5614
2025-02-14 13:04:57,194 [INFO] DocID=548, Float=0.5650, Int8=0.5364, Diff=0.0286
2025-02-14 13:04:57,194 [INFO] DocID=47, Float=0.5152, Int8=0.5802, Diff=0.0650
2025-02-14 13:04:57,194 [INFO] DocID=52, Float=0.5192, Int8=0.5614, Diff=0.0422
2025-02-14 13:04:57,194 [INFO] DocID=309 => ONLY in Int8 top10, Score=0.5833
2025-02-14 13:04:57,194 [INFO] DocID=822 => ONLY in Float top10, Score=0.4944
2025-02-14 13:04:57,194 [INFO] DocID=827, Float=0.5273, Int8=0.5458, Diff=0.0185
2025-02-14 13:04:57,194 [INFO] DocID=832, Float=0.4977, Int8=0.5708, Diff=0.0731
2025-02-14 13:04:57,194 [INFO] DocID=581, Float=0.4926, Int8=0.5614, Diff=0.0688
2025-02-14 13:04:57,194 [INFO] DocID=327 => ONLY in Float top10, Score=0.5419
2025-02-14 13:04:57,194 [INFO] DocID=839, Float=0.5096, Int8=0.5646, Diff=0.0550
2025-02-14 13:04:57,194 [INFO] DocID=590, Float=0.5375, Int8=0.5458, Diff=0.0083
2025-02-14 13:04:57,194 [INFO] DocID=851, Float=0.5896, Int8=0.4957, Diff=0.0939
2025-02-14 13:04:57,194 [INFO] DocID=597, Float=0.5224, Int8=0.5458, Diff=0.0233
2025-02-14 13:04:57,194 [INFO] DocID=858 => ONLY in Float top10, Score=0.4967
2025-02-14 13:04:57,194 [INFO] DocID=604 => ONLY in Int8 top10, Score=0.5802
2025-02-14 13:04:57,194 [INFO] DocID=864, Float=0.5196, Int8=0.5364, Diff=0.0168
2025-02-14 13:04:57,195 [INFO] DocID=612 => ONLY in Int8 top10, Score=0.5677
2025-02-14 13:04:57,195 [INFO] DocID=871, Float=0.5176, Int8=0.5426, Diff=0.0251
2025-02-14 13:04:57,195 [INFO] DocID=874 => ONLY in Float top10, Score=0.5144
2025-02-14 13:04:57,195 [INFO] DocID=620, Float=0.5532, Int8=0.4926, Diff=0.0606
2025-02-14 13:04:57,195 [INFO] DocID=110 => ONLY in Float top10, Score=0.4950
2025-02-14 13:04:57,195 [INFO] DocID=627 => ONLY in Float top10, Score=0.5056
2025-02-14 13:04:57,195 [INFO] DocID=375, Float=0.5326, Int8=0.5552, Diff=0.0226
2025-02-14 13:04:57,195 [INFO] DocID=377 => ONLY in Int8 top10, Score=0.5614
2025-02-14 13:04:57,195 [INFO] DocID=124, Float=0.5328, Int8=0.5176, Diff=0.0151
2025-02-14 13:04:57,195 [INFO] DocID=381, Float=0.5095, Int8=0.5865, Diff=0.0770
2025-02-14 13:04:57,195 [INFO] DocID=643, Float=0.5489, Int8=0.5583, Diff=0.0094
2025-02-14 13:04:57,195 [INFO] DocID=653, Float=0.5328, Int8=0.5739, Diff=0.0411
2025-02-14 13:04:57,195 [INFO] DocID=145 => ONLY in Float top10, Score=0.4940
2025-02-14 13:04:57,195 [INFO] DocID=659, Float=0.5761, Int8=0.5301, Diff=0.0460
2025-02-14 13:04:57,195 [INFO] DocID=661 => ONLY in Int8 top10, Score=0.5802
2025-02-14 13:04:57,195 [INFO] DocID=670 => ONLY in Float top10, Score=0.5046
2025-02-14 13:04:57,195 [INFO] DocID=675 => ONLY in Float top10, Score=0.5008
2025-02-14 13:04:57,195 [INFO] DocID=429 => ONLY in Int8 top10, Score=0.5865
2025-02-14 13:04:57,195 [INFO] DocID=430, Float=0.5447, Int8=0.5207, Diff=0.0239
2025-02-14 13:04:57,195 [INFO] DocID=177, Float=0.5285, Int8=0.5646, Diff=0.0360
2025-02-14 13:04:57,195 [INFO] DocID=434, Float=0.5178, Int8=0.5270, Diff=0.0092
2025-02-14 13:04:57,195 [INFO] DocID=695, Float=0.5411, Int8=0.5113, Diff=0.0298
2025-02-14 13:04:57,195 [INFO] DocID=952, Float=0.5742, Int8=0.5207, Diff=0.0535
2025-02-14 13:04:57,195 [INFO] DocID=440, Float=0.5076, Int8=0.5677, Diff=0.0601
2025-02-14 13:04:57,195 [INFO] DocID=703, Float=0.5356, Int8=0.5176, Diff=0.0180
2025-02-14 13:04:57,195 [INFO] DocID=197 => ONLY in Int8 top10, Score=0.5833
2025-02-14 13:04:57,195 [INFO] DocID=463, Float=0.5292, Int8=0.5489, Diff=0.0197
2025-02-14 13:04:57,195 [INFO] DocID=208, Float=0.5043, Int8=0.5426, Diff=0.0383
2025-02-14 13:04:57,195 [INFO] DocID=473 => ONLY in Int8 top10, Score=0.5833
2025-02-14 13:04:57,195 [INFO] DocID=219, Float=0.5012, Int8=0.5677, Diff=0.0664
2025-02-14 13:04:57,195 [INFO] DocID=734 => ONLY in Int8 top10, Score=0.5708
2025-02-14 13:04:57,195 [INFO] DocID=991 => ONLY in Int8 top10, Score=0.5833
2025-02-14 13:04:57,196 [INFO] DocID=485, Float=0.5231, Int8=0.5176, Diff=0.0055
2025-02-14 13:04:57,196 [INFO] DocID=240, Float=0.5209, Int8=0.5802, Diff=0.0593
2025-02-14 13:04:57,196 [INFO] DocID=242, Float=0.5616, Int8=0.5489, Diff=0.0127
2025-02-14 13:04:57,196 [INFO] DocID=499 => ONLY in Float top10, Score=0.5142
2025-02-14 13:04:57,196 [INFO] DocID=503 => ONLY in Float top10, Score=0.5026
2025-02-14 13:04:57,196 [INFO] DocID=764 => ONLY in Int8 top10, Score=0.5458
2025-02-14 13:04:57,196 [INFO] DocID=510, Float=0.5468, Int8=0.5239, Diff=0.0230
2025-02-14 13:04:57,196 [INFO] 
Differences Stats: avg=0.0369, min=0.0029, max=0.0939
2025-02-14 13:04:57,196 [INFO] 
=== Summary of Performance Metrics ===
2025-02-14 13:04:57,196 [INFO] Float DB Build Time:    26.69 sec
2025-02-14 13:04:57,196 [INFO] Int8 DB Build Time:     14.54 sec (45.52% faster)
2025-02-14 13:04:57,196 [INFO] Enhanced DB Build Time: 14.45 sec
2025-02-14 13:04:57,196 [INFO] Float DB Search Time:   0.52 sec
2025-02-14 13:04:57,196 [INFO] Int8 DB Search Time:    0.51 sec (2.73% faster)
2025-02-14 13:04:57,196 [INFO] Enhanced DB Search Time:0.51 sec
2025-02-14 13:04:57,196 [INFO] Float DB Size:          4.04 MB
2025-02-14 13:04:57,196 [INFO] Int8 DB Size:           1.38 MB (65.94% smaller)
2025-02-14 13:04:57,196 [INFO] Enhanced DB Size:       1.38 MB
2025-02-14 13:04:57,196 [INFO] Rerank Time:            1.25 sec

```

## License

© 2025 Constantine Vassilev. All rights reserved.