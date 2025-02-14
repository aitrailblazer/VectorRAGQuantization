CohereEnhancedVectorDB

An enhanced Cohere-based vector database that uses HTTP requests to the Cohere embed endpoint. It leverages multiple embedding types for efficient document indexing and retrieval using FAISS and RocksDict.

Overview

CohereEnhancedVectorDB is designed for fast, accurate document retrieval while minimizing storage overhead. It retrieves quantized embeddings (int8, ubinary, and float) from Cohere’s API and uses a multi-phase search process. The system stores document texts alongside their quantized embeddings in RocksDict and builds a binary index with FAISS, enabling rapid approximate nearest-neighbor searches based on binary representations.

Key Features
	•	HTTP-Based Embedding Generation
Communicates with the Cohere embed endpoint using HTTP POST requests. The endpoint, model name, and desired embedding types are configured via environment variables.
	•	Multiple Embedding Types
Retrieves int8 and ubinary embeddings for document indexing. For search queries, a float representation is derived from the int8 embeddings to support refined similarity computations.
	•	Efficient Indexing with FAISS
Uses FAISS to build a binary index from ubinary embeddings, facilitating fast retrieval based on Hamming distance.
	•	Document Storage with RocksDict
Stores each document’s text and its int8 embedding in RocksDict, ensuring that detailed document data is available for rescoring during search.
	•	Three-Phase Search Process
	•	Phase I – Fast Retrieval: Quickly retrieves an initial candidate set using a binary search with the ubinary representation of the query.
	•	Phase II – Dot-Product Rescoring: Rescores candidates by computing a dot-product between the query’s float representation and the unpacked binary document embeddings.
	•	Phase III – Cosine Similarity Rescoring: Finalizes the ranking by computing cosine similarity between the query’s float representation and the stored int8 embeddings.

Configuration

Before using CohereEnhancedVectorDB, set the following environment variables:
	•	COHERE_EMBED_ENDPOINT: The URL for the Cohere embedding API.
	•	COHERE_EMBED_KEY: The API key for authenticating with the Cohere service.

Workflow
	1.	Initialization:
The system creates or loads a configuration file containing the model name and embedding dimension, initializes a FAISS binary index, and sets up RocksDict for document storage.
	2.	Document Indexing:
Documents are processed in batches. For each batch, the system requests int8 and ubinary embeddings from the Cohere API. Ubinary embeddings are added to the FAISS index for fast search, while int8 embeddings and corresponding document texts are stored in RocksDict.
	3.	Search Process:
	•	Query Processing: The system retrieves query embeddings from Cohere (including a float representation and ubinary embedding).
	•	Phase I: A fast binary search using the ubinary query embedding retrieves a set of candidate documents.
	•	Phase II: Candidates are rescored via a dot-product between the query’s float representation and the unpacked binary embeddings from FAISS.
	•	Phase III: Final rescoring is performed by computing cosine similarity between the query’s float representation and the stored int8 embeddings, resulting in a ranked list of matching documents.

Logging and Debugging

CohereEnhancedVectorDB uses Python’s logging framework to provide detailed operational insights. Debug messages—including API request payload details—are available when the logging level is set to DEBUG; by default, the system logs at the INFO level.

Summary

CohereEnhancedVectorDB combines the efficiency of quantized embeddings and binary indexing with a robust multi-phase search process. Its modular design, compact storage, and advanced rescoring techniques make it an ideal solution for high-performance document retrieval in large-scale applications.