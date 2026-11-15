<!-- You are a Senior AI Engineer and Software Architect at Anthropic/OpenAI.

I want to build a production-quality enterprise GenAI application from scratch.

Project Name:
NexusAI — Enterprise Agentic Knowledge Platform

Goal:
Build an enterprise-grade Agentic RAG platform that allows users to upload and query multiple knowledge sources such as PDFs, DOCX files, CSV files, websites, GitHub repositories and knowledge graphs through natural language.

The project should be modular, production-ready, and follow clean architecture.

Main Features

• Multi-source document ingestion
• PDF, DOCX, CSV parsing
• Website crawling
• GitHub repository ingestion
• GraphRAG using Neo4j
• Hybrid RAG (Graph + Vector Search)
• Semantic Search
• Hybrid Search (Vector + BM25)
• Query Rewriting
• Context Compression
• Reranking
• Conversation Memory
• Source Citations
• Tavily Web Search
• MCP Tool Calling
• Authentication
• Docker Deployment
• REST API
• Modern responsive UI

Technology Stack

Frontend
- Next.js
- TypeScript
- TailwindCSS
- ShadCN UI

Backend
- FastAPI
- Python

LLM
- Gemini 2.5 Flash

Framework
- LangChain

Embeddings
- BAAI BGE-M3

Vector Database
- Qdrant

Knowledge Graph
- Neo4j

GraphRAG
- GraphRAG pipeline

Observability/ Evals
- LangSmith

Search
- Tavily API

Document Parsing
- Docling

OCR
- PaddleOCR

Deployment
- Docker
- Vercel


Development Requirements

Do NOT generate the entire project at once.

Act as a senior software architect.

First design the complete production architecture.

Then break the project into milestones.

Each milestone should be independently runnable.

For every milestone provide

• Folder structure
• Architecture diagram
• File list
• APIs
• Database schema
• Environment variables
• Installation steps
• Required packages
• Best practices
• Common bugs
• Testing checklist

After completing one milestone, stop and wait for my confirmation before generating the next.

The project should be modular, scalable and easy to maintain.
NexusAI

                User
                 |
                 ↓
            Next.js UI
                 |
                 ↓
             FastAPI
                 |
    --------------------------------
    |              |               |
 Gemini API      Qdrant          Neo4j
                 |               |
          Vector Retrieval   Graph Reasoning
                 |
                 ↓
          LangChain Pipeline
                 |
                 ↓
            Final Answer

                 |
                 ↓
             LangSmith
             Evaluation

🚀Start NEXUS AI

Terminal 1 — Start NexusAI Backend
cd /Users/adrikamandal/Downloads/NexusAI/apps/backend
source .venv/bin/activate
uvicorn nexusai.main:app --reload


http://127.0.0.1:8000/docs  - Swagger UI Backend 

Terminal 2 — Start NexusAI Frontend

cd /Users/adrikamandal/Downloads/NexusAI/apps/frontend
npm run dev
Frontend:
http://localhost:3001

IDE Terminal — Start Databases
docker compose up qdrant neo4j
Qdrant : http://localhost:6333/dashboard
Neo4j : http://localhost:7474/
 

