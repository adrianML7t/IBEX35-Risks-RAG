# 📚 RAG with LangChain, ChromaDB, and Ollama

This project implements a local **RAG (Retrieval-Augmented Generation)** system. The system indexes documents into a vector database and uses an LLM (Llama 3.2) to answer questions based strictly on the content of those files.

## 🚀 Features

* **Document Processing:** Loads and splits PDF files from the `Informes` folder.
* **Vector Database:** Uses **ChromaDB** to persistently store embeddings.
* **Multilingual Embeddings:** Uses `paraphrase-multilingual-mpnet-base-v2` for precise semantic search in Spanish and English.
* **Local Inference:** Uses **Ollama** with the `llama3.2` model, ensuring privacy and offline execution.
