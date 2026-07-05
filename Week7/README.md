# Document Question Answering System using RAG

## Overview
This project implements a Retrieval-Augmented Generation system that answers user questions based on uploaded PDF documents.

## Objective
The objective is to build a document-based question answering system using retrieval and generation.

## Features
- Upload PDF documents
- Extract text from PDF
- Split text into smaller chunks
- Create embeddings
- Store embeddings in Chroma vector database
- Retrieve relevant document chunks
- Generate answers using Gemini LLM

## Technologies Used
- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Gemini API

## Workflow
1. Upload PDF
2. Load document text
3. Split text into chunks
4. Convert chunks into embeddings
5. Store vectors in ChromaDB
6. Accept user query
7. Retrieve relevant chunks
8. Generate final answer using LLM

## Key Learning
This project helped me understand how RAG systems combine retrieval and generation to answer questions from private or custom documents.

## Conclusion
The system successfully answers questions based on uploaded documents and demonstrates the working of a basic RAG pipeline.