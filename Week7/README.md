# Week 7 - RAG Document Question Answering System

## Project Title

**Document Question Answering System using Retrieval-Augmented Generation**

---

## Project Overview

This project is based on the concept of **Retrieval-Augmented Generation**, commonly known as **RAG**. The main objective of this project is to build a document-based question answering system where a user can upload a PDF file and ask questions related to the content of that document.

In traditional language models, the model generates answers based only on the knowledge it has learned during training. However, such models may not know the content of a newly uploaded document. To solve this problem, RAG is used.

In this project, the uploaded PDF document acts as an external knowledge source. The system extracts the content of the PDF, divides it into smaller chunks, converts those chunks into numerical embeddings, stores them in a vector database, and retrieves the most relevant chunks when a user asks a question.

The final answer is generated from the retrieved document context. This helps the system provide answers that are more relevant to the uploaded document.

---

## Objective of the Project

The objective of this project is to understand and implement the basic workflow of a RAG-based application.

The project focuses on:

- Loading PDF documents
- Extracting text from PDF files
- Splitting large text into smaller chunks
- Creating embeddings from text chunks
- Storing embeddings in a vector database
- Retrieving relevant chunks based on user queries
- Displaying answers from the retrieved context
- Building a simple user interface using Streamlit

---

## Why This Project is Useful

Large Language Models are powerful, but they have some limitations:

1. They may not know private or newly uploaded data.
2. They may generate incorrect information if context is not provided.
3. They cannot automatically access the content of a PDF unless it is processed.
4. They may hallucinate if asked about unknown information.

RAG helps solve these problems by giving the model access to external documents during the answering process.

In this project, the PDF uploaded by the user becomes the knowledge base. The system retrieves relevant information from that PDF and uses it to answer user questions.

---

## Features

This project includes the following features:

- User-friendly Streamlit interface
- PDF file upload option
- Automatic PDF text extraction
- Text splitting into smaller chunks
- Local embedding generation using HuggingFace Sentence Transformers
- Vector storage using ChromaDB
- Similarity-based retrieval of relevant chunks
- Question answering from retrieved context
- Display of retrieved chunks for better transparency
- No dependency on paid API for embeddings
- Stable local RAG pipeline for demonstration

---

## Technologies Used

The following technologies and libraries are used in this project:

### Python

Python is used as the main programming language because it provides strong support for machine learning, NLP, and web application development.

### Streamlit

Streamlit is used to create the web interface of the application. It allows users to upload PDF files and ask questions directly from the browser.

### LangChain

LangChain is used to build the RAG pipeline. It provides components for document loading, text splitting, embeddings, vector stores, and retrieval.

### PyPDFLoader

PyPDFLoader is used to load and extract text from PDF documents.

### RecursiveCharacterTextSplitter

RecursiveCharacterTextSplitter is used to split large PDF content into smaller and meaningful chunks.

### HuggingFace Embeddings

HuggingFace embeddings are used to convert text chunks into numerical vector representations.

### Sentence Transformers

The Sentence Transformer model is used to generate semantic embeddings from text.

The model used in this project is:

```text
sentence-transformers/all-MiniLM-L6-v2
