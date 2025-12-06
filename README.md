# 📚 RAG con LangChain, ChromaDB y Ollama

Este proyecto implementa un sistema de **RAG (Retrieval-Augmented Generation)** local. El sistema indexa los documentos en una base de datos vectorial y utiliza un LLM (Llama 3.2) para responder preguntas basándose estrictamente en el contenido de esos archivos.

## 🚀 Características

* **Procesamiento de Documentos:** Carga y divide archivos PDF de la carpeta Informes`.
* **Base de Datos Vectorial:** Utiliza **ChromaDB** para almacenar los embeddings de forma persistente.
* **Embeddings Multilingües:** Usa `paraphrase-multilingual-mpnet-base-v2` para una búsqueda semántica precisa en español e inglés.
* **Inferencia Local:** Utiliza **Ollama** con el modelo `llama3.2`, garantizando privacidad y ejecución offline.

## 📋 Requisitos Previos

1.  **Python 3.10+** instalado.
2.  **Ollama** instalado y ejecutándose.
    * Descárgalo en [ollama.com](https://ollama.com).
    * Descarga el modelo necesario ejecutando en tu terminal:
        ```bash
        ollama pull llama3.2
        ```

## 🛠️ Instalación

1.  Clona este repositorio o descarga los archivos.
2.  Crea un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Estructura del Proyecto

Antes de ejecutar, asegúrate de tener la siguiente estructura:

```text
.
├── create_database.py    # Script para generar/actualizar la base de datos
├── query_database.py     # Script para realizar consultas
├── Informes/             # Carpeta con los documentos financieros
└── chroma_db/            # Se generará automáticamente aquí la base de datos
