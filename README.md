## Grok-Quantized-RAG-Navigator

pip install -r dependencies.txt

```bash
python main.py
```

# Grok-Quantized-RAG-Navigator

A project comparing local and global quantization strategies for vector databases, focusing on performance, precision, efficiency, and disk storage.

## Overview

This repository contains implementations and analyses of two quantization methods for vector databases:

- **Local Quantization** (`VectorDBInt16`): Quantizes each document's embeddings independently.
- **Global Quantization** (`VectorDBInt16Global`): Applies a single quantization limit to all embeddings.

### Performance Analysis

#### Execution Summary:

- **Documents Loaded:** 1000 from `Generated_AI_Examples.csv`.
- **Query:** "Artificial intelligence is transforming industries."

**Comparison Table:**

| Feature                     | Local Quantization (VectorDBInt16) | Global Quantization (VectorDBInt16Global) |
|-----------------------------|------------------------------------|-------------------------------------------|
| **Indexing Time**           | 40 seconds (~24.74 docs/sec)       | 42 seconds (~23.42 docs/sec)              |
| **Max Score Diff**          | 0.0055%                            | 0.0021%                                   |
| **Min Score Diff**          | 0.0044%                            | 0.0002%                                   |
| **Avg Percentage Diff**     | 0.0048%                            | 0.0011%                                   |
| **Median Percentage Diff**  | 0.0047%                            | 0.0011%                                   |
| **Storage Overhead**        | Higher (per-doc `min_max` values)  | Lower (uniform scaling)                   |
| **Precision**               | High for individual documents      | Uniform across all documents              |
| **Complexity**              | More complex due to document-specific scaling | Simpler setup and maintenance |

### Algorithm Descriptions:

#### Local Quantization (VectorDBInt16):

- **Method**: Each document's embedding is quantized independently, scaling based on its own min and max values. This retains the original distribution but requires storing `min_max` values for dequantization.

- **Pros**:
  - High precision for documents with unique value distributions.
  - Better handling of outliers within individual documents.

- **Cons**:
  - Higher storage requirements due to additional metadata.
  - More computational overhead for scaling and dequantization during searches.

#### Global Quantization (VectorDBInt16Global):

- **Method**: Uses a single global clipping limit for all embeddings, applying uniform scaling across documents. This simplifies data management but might not handle outliers as well for all documents.

- **Pros**:
  - Lower storage overhead since no per-document scaling data is needed.
  - Uniform quantization reduces complexity, leading to consistent performance.

- **Cons**:
  - Might lose precision for documents with values outside the common range.

### Disk Databases

Both implementations leverage disk-based storage using:
- **FAISS** for efficient similarity search on disk.
- **RocksDB** (via `Rdict`) for persistent document storage.

**Advantages of Disk Databases over In-Memory (Float32):**

- **Memory Efficiency**: Quantized embeddings (Int16) significantly reduce memory usage compared to Float32, allowing larger datasets to be stored and processed with less RAM.
  
- **Disk Space Savings**: Int16 quantization halves the storage requirement per embedding vector compared to Float32, making it feasible to store vast amounts of data on disk without filling up storage.

- **Faster Index Loading**: Since the index can be stored on disk, loading times for large datasets are reduced, and the system can handle restarts or scale across sessions without losing data.

- **Scalability**: Disk storage enables scaling to datasets that are too large for RAM, providing a solution for big data scenarios in similarity searches.

- **Precision vs. Performance**: While there's a marginal loss in precision compared to Float32, the quantization strategies offer a balance where the precision loss is negligible for many applications, providing a trade-off for speed and storage efficiency.

### Use Cases:

#### Local Quantization (VectorDBInt16):

- **Specialized Content**: Ideal for content with varied embedding value ranges, like scientific articles or niche topics.
- **High Precision Requirement**: Critical for applications needing precise similarity scores, e.g., medical diagnostics or personalized recommendations.
- **Small to Medium Datasets**: Where storage isn't a concern, and precision is prioritized.

#### Global Quantization (VectorDBInt16Global):

- **Generalized Content**: Best for datasets with uniform content, like general news or broad forums.
- **Large Scale Applications**: Perfect for scenarios with millions of documents where memory efficiency is key.
- **Real-time Systems**: Where speed and uniform performance across documents are more important than individual precision.

 
### Visualizations

Below are examples of the visualizations generated by this project:

#### Local Quantization

**Int16**

- **Score Comparison:**
  <img src="img/Int16_scores_comparison.png" alt="Int16 Score Comparison" width="800" height="600">

- **Side by Side Score Comparison:**
  <img src="img/Int16_Side_By_Side_scores_comparison.png" alt="Int16 Side By Side Score Comparison" width="800" height="600">

- **Percentage Differences:**
  <img src="img/Int16_percentage_diffs.png" alt="Int16 Percentage Differences" width="800" height="600">

- **Side by Side Percentage Differences:**
  <img src="img/Int16_Side_By_Side_percentage_diffs.png" alt="Int16 Side By Side Percentage Differences" width="800" height="600">

**Int8**

- **Score Comparison:**
  <img src="img/Int8_scores_comparison.png" alt="Int8 Score Comparison" width="800" height="600">

- **Side by Side Score Comparison:**
  <img src="img/Int8_Side_By_Side_scores_comparison.png" alt="Int8 Side By Side Score Comparison" width="800" height="600">

- **Percentage Differences:**
  <img src="img/Int8_percentage_diffs.png" alt="Int8 Percentage Differences" width="800" height="600">

- **Side by Side Percentage Differences:**
  <img src="img/Int8_Side_By_Side_percentage_diffs.png" alt="Int8 Side By Side Percentage Differences" width="800" height="600">

**Int4**

- **Score Comparison:**
  <img src="img/Int4_scores_comparison.png" alt="Int4 Score Comparison" width="800" height="600">

- **Side by Side Score Comparison:**
  <img src="img/Int4_Side_By_Side_scores_comparison.png" alt="Int4 Side By Side Score Comparison" width="800" height="600">

- **Percentage Differences:**
  <img src="img/Int4_percentage_diffs.png" alt="Int4 Percentage Differences" width="800" height="600">

- **Side by Side Percentage Differences:**
  <img src="img/Int4_Side_By_Side_percentage_diffs.png" alt="Int4 Side By Side Percentage Differences" width="800" height="600">

#### Global Quantization

**Int16Global**

- **Score Comparison:**
  <img src="img/Int16Global_scores_comparison.png" alt="Int16Global Score Comparison" width="800" height="600">

- **Side by Side Score Comparison:**
  <img src="img/Int16Global_Side_By_Side_scores_comparison.png" alt="Int16Global Side By Side Score Comparison" width="800" height="600">

- **Percentage Differences:**
  <img src="img/Int16Global_percentage_diffs.png" alt="Int16Global Percentage Differences" width="800" height="600">

- **Side by Side Percentage Differences:**
  <img src="img/Int16Global_Side_By_Side_percentage_diffs.png" alt="Int16Global Side By Side Percentage Differences" width="800" height="600">

**Int8Global**

- **Score Comparison:**
  <img src="img/Int8Global_scores_comparison.png" alt="Int8Global Score Comparison" width="800" height="600">

- **Side by Side Score Comparison:**
  <img src="img/Int8Global_Side_By_Side_scores_comparison.png" alt="Int8Global Side By Side Score Comparison" width="800" height="600">

- **Percentage Differences:**
  <img src="img/Int8Global_percentage_diffs.png" alt="Int8Global Percentage Differences" width="800" height="600">

- **Side by Side Percentage Differences:**
  <img src="img/Int8Global_Side_By_Side_percentage_diffs.png" alt="Int8Global Side By Side Percentage Differences" width="800" height="600">

**Int4Global**

- **Score Comparison:**
  <img src="img/Int4Global_scores_comparison.png" alt="Int4Global Score Comparison" width="800" height="600">

- **Side by Side Score Comparison:**
  <img src="img/Int4Global_Side_By_Side_scores_comparison.png" alt="Int4Global Side By Side Score Comparison" width="800" height="600">

- **Percentage Differences:**
  <img src="img/Int4Global_percentage_diffs.png" alt="Int4Global Percentage Differences" width="800" height="600">

- **Side by Side Percentage Differences:**
  <img src="img/Int4Global_Side_By_Side_percentage_diffs.png" alt="Int4Global Side By Side Percentage Differences" width="800" height="600">

#### Combined Visualization

- **All Methods Percentage Differences:**
  <img src="img/percentage_diffs_comparison.png" alt="All Methods Percentage Differences Comparison" width="800" height="600">

These plots are automatically generated when you run the respective quantization methods. They help in visually assessing how quantization affects document similarity scores and the distribution of these effects across documents. The combined visualization provides a direct comparison of quantization accuracy across all methods.

### Conclusion

Choosing between local and global quantization depends on whether your application requires high precision for individual documents or prioritizes efficiency and uniformity across large datasets. Both approaches offer significant advantages over using Float32 embeddings in terms of storage, memory usage, and scalability.
## Setup & Running

- **Dependencies**: Ensure you have Python installed with necessary libraries like `numpy`, `pandas`, `faiss`, `requests`, etc.
- **Running**: Use the script `bld.bash` to run the main analysis script `main.py`:
