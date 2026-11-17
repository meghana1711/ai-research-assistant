# AI Research Assistant
A domain-specific **Retrieval-Augmented Generation (RAG) research assistant** for answering questions from AI/ML research papers. 
The project focuses on building and evaluating a Langchain based RAG pipeline using the **QASPER research-paper question-answering benchmark**, with particular attention to whether improved retrieval leads to better final answers.

## Project Overview
The system combines :
- Document ingestion with Docling
- Dense retrieval
- Hybrid search with Qdrant
- Cross encoder reranking
- Local LLM generation using Qwen3 4B through Ollama
- Answers with retrieved-source citations
- Retrieval and answer quality evaluation using QASPER

## Pipeline
Research Paper
      ↓
Document Processing
      ↓
Dense + Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Top Relevant Evidence
      ↓
Qwen3 4B
      ↓
Answer + Sources

For uploaded PDFs, Docling is used for document processing. QASPER provides already-structured research-paper text and is used separately for benchmarking.

## Evaluation
The project compares a dense-retrieval baseline with hybrid retrieval and cross-encoder reranking.
Retrieval is measured using:
- Hit@K
- Recall@K
- MRR@10
Generated answers are evaluated against QASPER human reference answers using **Answer F1**.

## Installation
### 1. Clone the repository
```bash
git clone <your-repository-url>
cd ai-research-assistant
```
### 2. Install dependencies
```bash
pip install uv
uv sync
```
### 3. Install Ollama
Install Ollama from https://ollama.com and download Qwen3 4B:
```bash
ollama pull qwen3:4b
```
### 4. Environment configuration
Create a `.env` file:
```env
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434
```
## Run 
Test the RAG pipeline:
```bash
uv run python -m scripts.test_generate
```
Run generation evaluation:
```bash
uv run python -m evaluation.evaluate_generation
```

## Technologies
Python · LangChain · Docling · Qdrant · Sentence Transformers · FastEmbed/BM25 · Cross-Encoder · Ollama · Qwen3 4B · QASPER
