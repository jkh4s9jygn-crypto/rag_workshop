
# 🔍 Advanced RAG Pipeline with Hybrid Search and Re-Ranking

This repository contains a fully working, modular Python implementation of an advanced **Retrieval-Augmented Generation (RAG)** pipeline. Built using Google Colab and modern NLP libraries, the notebook demonstrates how to build a production-grade question-answering system over a product manual (`Aurora M2 Wireless Headphones`).

---

## 🚀 Key Features & Pipeline Architecture

The implementation moves beyond basic vector search by incorporating a multi-stage hybrid retrieval and cross-encoder re-ranking strategy to achieve high precision:

1. **Smart Text Chunking**:
* Compares fixed-size, recursive paragraph-based, and **section-based splitting**.


* Leverages regex to segment documentation logically by markdown headings (`##`).




2. **Metadata Tagging**: Automatically tags document chunks by content type (e.g., `troubleshooting`, `policy`, `specs`, `faq`) to enable filtered searches.


3. **Hybrid Search (Dense + Sparse)**:
* **Vector Search**: Uses `sentence-transformers` (`all-mpnet-base-v2`) and **FAISS** (`IndexFlatIP`) with normalized embeddings for semantic similarity.


* **Keyword Search**: Uses `rank-bm25` (BM25Okapi) with a customized tokenizer optimized to preserve product codes and error formats (e.g., `E-03`).


* **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse retrieval results smoothly.




4. **Cross-Encoder Re-Ranking**: Refines the initial candidate pool using a powerful cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to deeply evaluate query-chunk context pairs before generation.


5. **LLM Generation**: Connects to an OpenAI-compatible endpoint (`poolside/laguna-s-2.1`) to generate grounded, context-aware answers complete with mandatory section citations.



---

## 📦 Tech Stack & Dependencies

The pipeline relies on the following open-source Python libraries (installed via the notebook):

* `sentence-transformers` – For semantic embeddings and cross-encoder re-ranking.


* `faiss-cpu` – For efficient vector similarity search.


* `rank-bm25` – For traditional keyword-based BM25 matching.


* `openai` – For interfacing with the LLM generation backend.



---

## ⚙️ Quick Start

You can run this notebook directly in Google Colab or locally:

1. Clone the repository or download the `.ipynb` file.
2. Upload your target Markdown document (e.g., `product_manual_1.md`) when prompted by the script.


3. Run the cells sequentially to initialize the index, test various search configurations, and execute sample RAG queries.



### Example Query Evaluation

The pipeline successfully handles complex queries, handles domain-specific error codes, and rejects out-of-scope questions cleanly:

* *Q: What is the battery life with noise cancellation on?* $\rightarrow$ Retrieves specifications and breakdown details.


* *Q: What does error E-04 mean?* $\rightarrow$ Accurately isolates troubleshooting logs.


* *Q: What is the wifi password for the headphones?* $\rightarrow$ Correctly triggers fallback behavior: *"I could not find this in the manual."*
