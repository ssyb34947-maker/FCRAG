# Backend module for RAG system

This module provides a FastAPI-based RESTful API for the RAG system with CRUD operations.

## Features

- Create, Read, Update, Delete documents
- Query documents by text similarity
- Health check endpoint

## API Endpoints

- `POST /documents/` - Create a new document
- `GET /documents/{doc_id}` - Retrieve a document by ID
- `PUT /documents/{doc_id}` - Update a document by ID
- `DELETE /documents/{doc_id}` - Delete a document by ID
- `POST /query/` - Query documents by text similarity
- `GET /health` - Health check

## Running the service

```bash
uvicorn backend.main:app --reload
```