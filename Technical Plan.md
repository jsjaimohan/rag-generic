# MiniRAG + Qwen3 8B + BGE-M3
# Complete Technical Architecture & Implementation Guide

## Version
1.0

---

# Table of Contents

1. Executive Summary
2. Final Architecture
3. Technology Stack
4. System Design Principles
5. End-to-End Data Flow
6. Development Environment Setup
7. Local Development Architecture
8. Production Architecture
9. Website Crawling Strategy
10. HTML Cleaning Pipeline
11. HTML → Markdown Conversion
12. Markdown Structure Standards
13. AI Enrichment Pipeline
14. AI-Generated Metadata
15. Semantic Chunking Strategy
16. Relationship Generation During Ingestion
17. GraphRAG Concepts with MiniRAG
18. Embedding Pipeline
19. Re-Ranking Pipeline
20. Hybrid Retrieval Strategy
21. Vector Database Design
22. Qdrant Collection Design
23. MiniRAG Integration
24. Prompt Engineering Strategy
25. Response Generation Pipeline
26. Caching Strategy
27. Evaluation & Testing
28. Monitoring & Observability
29. Scaling to Multi-GPU Infrastructure
30. Security Considerations
31. Folder Structure
32. API Design
33. Recommended Models
34. Implementation Roadmap
35. Architecture Phase 2 — Graph-aware upgrade (deferred)
36. Future Enhancements
37. Final Recommendations

---

# 1. Executive Summary

This document describes the complete architecture and implementation plan for building a production-grade RAG chatbot using:

- MiniRAG
- Qwen3 8B Instruct
- BGE-M3 embeddings
- Qdrant vector database
- Markdown-first ingestion
- Semantic chunking
- GraphRAG-style relationship mapping
- Re-ranking pipeline

The system is optimized for:

- Website documentation
- API documentation
- Technical knowledge bases
- Tables and structured data
- FAQs
- Enterprise support portals

The architecture prioritizes:

- High retrieval precision
- Grounded answers
- Self-hosting
- Low hallucination rate
- Scalable production deployment
- Local development on Apple Silicon

---

# 2. Final Architecture

```text
Website Sources
      ↓
Crawler Layer
      ↓
HTML Cleaning Pipeline
      ↓
Markdown Conversion
      ↓
AI Enrichment Pipeline
      ↓
Semantic Chunking
      ↓
Relationship Generation
      ↓
Embedding Pipeline
      ↓
Qdrant Vector Store
      ↓
MiniRAG Retrieval
      ↓
Re-Ranking Pipeline
      ↓
Qwen3 8B Instruct
      ↓
Final Response
```

---

# 3. Technology Stack

| Layer | Technology |
|---|---|
| LLM | Qwen3 8B Instruct |
| Embeddings | BGE-M3 |
| Re-ranker | bge-reranker-v2 |
| RAG Framework | MiniRAG |
| Vector Database | Qdrant |
| Runtime | LM Studio (Local) |
| Crawler | Playwright |
| HTML Extraction | trafilatura |
| Markdown Conversion | markdownify |
| Backend | Python + FastAPI |
| Queue | Redis / RabbitMQ |
| Storage | Markdown Files + Qdrant |
| Deployment | Docker |
| Production Inference | vLLM |

---

# 4. System Design Principles

## Core Principles

### 1. Retrieval Quality > Model Size

The architecture prioritizes:

- chunk quality
- metadata quality
- retrieval precision
- reranking

instead of excessively large LLMs.

---

### 2. Preserve Original Content

Never replace original documentation entirely with summaries.

Correct approach:

```text
Original Content
+ AI Summary
+ Metadata
+ Relationships
```

---

### 3. Markdown-First Architecture

Markdown provides:

- semantic structure
- hierarchy preservation
- LLM friendliness
- portability
- readability

---

### 4. Graph-Aware Retrieval

The system generates semantic relationships during ingestion.

Example:

```json
{
  "chunk_id": "oauth_pkce",
  "related_chunks": [
    "oauth_overview",
    "jwt_validation",
    "token_expiry"
  ]
}
```

---

# 5. End-to-End Data Flow

## Complete Pipeline

```text
Website
  ↓
Playwright Crawl
  ↓
Extract Clean HTML
  ↓
Remove Noise
  ↓
Convert to Markdown
  ↓
Generate Metadata
  ↓
Generate Summaries
  ↓
Generate Relationships
  ↓
Semantic Chunking
  ↓
Generate Embeddings
  ↓
Store in Qdrant
  ↓
MiniRAG Retrieval
  ↓
Re-ranker
  ↓
Qwen3 Answer Generation
```

---

# 6. Development Environment Setup

## Local Development Machine

Recommended:

- Apple M2 Pro
- 32GB RAM preferred
- LM Studio
- Docker Desktop

---

## Install Core Components

### Python

```bash
brew install python
```

---

### Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

### Install Python Packages

```bash
pip install fastapi uvicorn qdrant-client sentence-transformers
pip install playwright markdownify trafilatura beautifulsoup4
pip install transformers torch accelerate
pip install rank-bm25
```

---

### Playwright Setup

```bash
playwright install
```

---

# 7. Local Development Architecture

```text
Frontend
   ↓
FastAPI Backend
   ↓
MiniRAG
   ├── Qdrant
   ├── Embedding Service
   ├── Re-ranker
   └── LM Studio API
```

---

# 8. Production Architecture

```text
Load Balancer
      ↓
API Gateway
      ↓
Application Layer
      ↓
MiniRAG Service
 ├── Retrieval Service
 ├── Embedding Service
 ├── Re-ranking Service
 └── Graph Relationship Service
      ↓
Qdrant Cluster
      ↓
vLLM GPU Cluster
```

---

# 9. Website Crawling Strategy

## Recommended Tool

### Playwright

Why:

- JavaScript rendering
- modern websites support
- stable automation
- easy authentication handling

---

## Crawl Strategy

### Crawl Scope

Include:

- documentation pages
- API pages
- FAQs
- support pages
- knowledge base articles

Exclude:

- login pages
- ads
- dashboards
- account settings
- search pages

---

## Example Crawl Logic

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url)
    html = page.content()
```

---

# 10. HTML Cleaning Pipeline

## Goals

Remove:

- navigation bars
- footers
- cookie banners
- ads
- unrelated sidebars
- scripts
- tracking elements

Preserve:

- headings
- paragraphs
- tables
- lists
- code blocks
- FAQs

---

## Recommended Library

### trafilatura

Why:

- excellent content extraction
- preserves structure
- removes boilerplate

---

## Example Cleaning Pipeline

```python
import trafilatura

clean_text = trafilatura.extract(html)
```

---

## Additional Cleaning Rules

### Remove Repeated Elements

Avoid embedding:

- repeated headers
- repeated footers
- repeated navigation

These pollute embeddings.

---

## Preserve Tables

DO NOT flatten tables.

Correct:

```markdown
| Token | Expiry |
|---|---|
| Access Token | 1 hour |
```

Incorrect:

```text
Access token expires in one hour.
```

---

# 11. HTML → Markdown Conversion

## Recommended Tool

### markdownify

---

## Example

```python
from markdownify import markdownify as md

markdown = md(html)
```

---

## Markdown Rules

Preserve:

- heading hierarchy
- tables
- bullet lists
- numbered lists
- code blocks
- hyperlinks

---

# 12. Markdown Structure Standards

## Recommended Markdown Template

````markdown
---
title: OAuth Authentication
url: https://example.com/auth
category: security
tags:
  - oauth2
  - pkce
summary:
  Authentication methods including OAuth2 and PKCE.
questions_answered:
  - Does the API support PKCE?
  - How are refresh tokens handled?
related:
  - token-lifecycle
  - jwt-validation
---

# OAuth Authentication

## Overview

## Supported Flows

### Authorization Code Flow

### PKCE Flow

## Token Lifecycle

| Token | Expiry |
|---|---|
| Access Token | 1 hour |
| Refresh Token | 30 days |

## FAQs

### Does the API support PKCE?

Yes.
````

---

# 13. AI Enrichment Pipeline

## Purpose

Generate:

- summaries
- tags
- entities
- relationships
- questions answered
- semantic metadata

---

## Why AI Enrichment Matters

Improves:

- retrieval precision
- conversational search
- reranking
- graph relationships

---

## Recommended Prompt

```text
Analyze the following markdown document.

Generate:
1. Summary
2. Tags
3. Key entities
4. Questions answered
5. Related concepts
6. Suggested relationships

Return JSON only.
```

---

# 14. AI-Generated Metadata

## Recommended Metadata Schema

```json
{
  "title": "",
  "url": "",
  "section": "",
  "summary": "",
  "tags": [],
  "entities": [],
  "questions_answered": [],
  "related_chunks": [],
  "document_type": "",
  "last_updated": ""
}
```

---

# 15. Semantic Chunking Strategy

## DO NOT Use Fixed Token Chunking Alone

Bad:

```text
Every 500 tokens
```

Good:

```text
Chunk by semantic sections
```

---

## Recommended Chunk Types

| Chunk Type | Purpose |
|---|---|
| summary | fast semantic retrieval |
| detailed | full explanations |
| faq | conversational retrieval |
| table | structured retrieval |
| code | API/code understanding |

---

## Recommended Chunk Size

| Type | Tokens |
|---|---|
| Small FAQ | 150–300 |
| Standard | 300–1000 |
| Large Technical | 1000–1500 |

---

## Chunk Overlap

Recommended:

```text
80–120 token overlap
```

---

## Semantic Chunking Example

```markdown
# OAuth Overview
```

becomes one chunk.

```markdown
## PKCE Flow
```

becomes another chunk.

---

# 16. Relationship Generation During Ingestion

## Why Relationships Matter

Relationships improve:

- GraphRAG retrieval
- multi-hop retrieval
- semantic navigation

---

## Relationship Types

| Type | Example |
|---|---|
| parent-child | section hierarchy |
| semantic | related concepts |
| entity | shared entities |
| faq linkage | question mappings |
| API linkage | endpoint relationships |

---

## Example Relationship Object

```json
{
  "chunk_id": "pkce_flow",
  "related_chunks": [
    "oauth_overview",
    "token_lifecycle",
    "jwt_validation"
  ],
  "relationship_type": "semantic"
}
```

---

## Relationship Generation Strategy

Generate relationships using:

- shared entities
- semantic similarity
- heading hierarchy
- hyperlinks
- AI enrichment

---

# 17. GraphRAG Concepts with MiniRAG

## Important Clarification

MiniRAG does not require a full graph database initially.

Graph behavior can be simulated through:

- metadata relationships
- semantic links
- related chunk references

---

## Recommended Initial Graph Structure

```json
{
  "chunk_id": "oauth_pkce",
  "parent_section": "authentication",
  "related_chunks": [
    "jwt_validation",
    "token_expiry"
  ],
  "entities": [
    "OAuth2",
    "PKCE",
    "JWT"
  ]
}
```

---

# 18. Embedding Pipeline

## Recommended Model

### BGE-M3

Reasons:

- excellent retrieval quality
- multilingual support
- strong semantic understanding
- GraphRAG compatibility
- hybrid retrieval support

---

## Embedding Generation Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
embedding = model.encode(text)
```

---

## Best Practices

### Embed:

- semantic chunks
- summaries
- FAQs
- tables

### Avoid Embedding:

- navigation
- repeated boilerplate
- empty sections

---

# 19. Re-Ranking Pipeline

## Recommended Model

### bge-reranker-v2

---

## Why Re-ranking Matters

Retrieval often returns:

- semantically similar chunks
- partially relevant chunks
- noisy chunks

The reranker improves precision.

---

## Retrieval Flow

```text
User Query
    ↓
Initial Vector Retrieval
    ↓
Top 20 Chunks
    ↓
Re-ranker
    ↓
Top 5 Final Chunks
    ↓
LLM Context
```

---

## Re-ranking Example

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3')
score = reranker.compute_score([query, document])
```

---

# 20. Hybrid Retrieval Strategy

## Recommended Retrieval Architecture

```text
Dense Search
   +
Sparse Search (BM25)
   +
Metadata Filtering
   +
Re-ranking
```

---

## Why Hybrid Retrieval Matters

Dense retrieval alone misses:

- exact keywords
- version numbers
- APIs
- technical terms

Sparse retrieval improves:

- exact matching
- identifiers
- configs

---

# 21. Vector Database Design

## Recommended Database

### Qdrant

Reasons:

- lightweight
- fast
- production-ready
- metadata filtering
- hybrid retrieval support

---

# 22. Qdrant Collection Design

## Suggested Payload Structure

```json
{
  "id": "chunk_001",
  "content": "",
  "summary": "",
  "title": "",
  "section": "",
  "tags": [],
  "entities": [],
  "questions_answered": [],
  "related_chunks": [],
  "document_type": "",
  "url": ""
}
```

---

## Collection Strategy

Recommended:

| Collection | Purpose |
|---|---|
| docs | documentation |
| faq | FAQs |
| api | API endpoints |
| tables | structured data |

---

# 23. MiniRAG Integration

## Responsibilities of MiniRAG

MiniRAG should handle:

- retrieval orchestration
- graph-aware retrieval
- chunk relationships
- context assembly

---

## MiniRAG Workflow

```text
User Query
   ↓
Retrieve Chunks
   ↓
Expand Related Chunks
   ↓
Apply Metadata Filters
   ↓
Re-rank
   ↓
Assemble Context
```

---

# 24. Prompt Engineering Strategy

## Core Prompt Structure

```text
You are a technical documentation assistant.

Answer ONLY using the provided context.

If information is missing, say:
"The documentation does not contain this information."
```

---

## Important Prompt Rules

- prohibit hallucinations
- force grounded answers
- prefer citations
- encourage concise responses

---

# 25. Response Generation Pipeline

```text
User Question
   ↓
Retrieve Chunks
   ↓
Expand Related Chunks
   ↓
Re-rank
   ↓
Assemble Prompt
   ↓
Qwen3 8B
   ↓
Generate Final Answer
```

---

# 26. Caching Strategy

## Recommended Caches

| Cache | Purpose |
|---|---|
| embedding cache | avoid recomputation |
| retrieval cache | repeated searches |
| LLM cache | repeated answers |
| markdown cache | avoid reconversion |

---

# 27. Evaluation & Testing

## Metrics

| Metric | Description |
|---|---|
| Recall@K | retrieval coverage |
| Precision@K | retrieval quality |
| Groundedness | hallucination control |
| Latency | response time |
| Answer Accuracy | correctness |

---

## Test Dataset

Create:

- expected questions
- expected answers
- expected source chunks

---

# 28. Monitoring & Observability

## Track

- retrieval latency
- reranking latency
- LLM latency
- hallucination rate
- failed queries
- low-confidence retrieval

---

# 29. Scaling to Multi-GPU Infrastructure

## Local Development

```text
LM Studio
```

---

## Production Upgrade

Replace with:

```text
vLLM
```

---

## Production GPU Architecture

```text
6 GPU Cluster
    ↓
vLLM Tensor Parallelism
    ↓
High Concurrency Inference
```

---

## Production Recommendations

| Component | Production |
|---|---|
| Inference | vLLM |
| Embeddings | dedicated embedding service |
| Vector DB | clustered Qdrant |
| Queue | RabbitMQ |
| Cache | Redis |

---

# 30. Security Considerations

## Important Controls

- sanitize HTML
- validate markdown
- rate-limit APIs
- secure vector database
- restrict admin ingestion
- remove secrets from crawled data

---

# 31. Folder Structure

```text
project/
├── backend/
├── crawler/
├── ingestion/
├── markdown/
├── embeddings/
├── reranker/
├── prompts/
├── qdrant/
├── minirag/
├── api/
├── frontend/
└── docker/
```

---

# 32. API Design

## Suggested APIs

| Endpoint | Purpose |
|---|---|
| /crawl | trigger crawl |
| /ingest | ingest markdown |
| /search | retrieve chunks |
| /chat | chatbot endpoint |
| /reindex | regenerate embeddings |

---

# 33. Recommended Models

## Final Recommended Stack

| Purpose | Model |
|---|---|
| LLM | Qwen3 8B Instruct |
| Embeddings | BGE-M3 |
| Re-ranker | bge-reranker-v2 |

---

## Quantization Recommendation

### Local Development

```text
Q4_K_M GGUF
```

---

# 34. Implementation Roadmap

# Phase 1 — Foundation

## Goals

- setup environment
- local Qdrant
- LM Studio integration
- basic crawl pipeline

---

# Phase 2 — Markdown Pipeline

## Goals

- HTML cleaning
- markdown conversion
- metadata generation

---

# Phase 3 — Chunking & Embeddings

## Goals

- semantic chunking
- embedding generation
- Qdrant ingestion

---

# Phase 4 — Retrieval

## Goals

- MiniRAG integration
- hybrid retrieval
- reranking

---

# Phase 5 — Chatbot

## Goals

- FastAPI APIs
- frontend chat UI
- prompt engineering

---

# Phase 6 — Evaluation

## Goals

- test datasets
- retrieval metrics
- hallucination analysis

## Evaluation modes (UI roadmap)

The frontend exposes three evaluation actions; implement and enable them in order:

1. **Text Keyword Evaluation** (current): runs `POST /evaluation/run` using keyword overlap over markdown files under `backend/data/processing`. No vectors or Qdrant in the retrieval path.
2. **Vector Evaluation** (planned): add backend route and Qdrant (or embedding) retrieval, then wire the UI button. Enable the button once vector search is the primary retriever for the same test-case JSON.
3. **RagGraph Evaluation** (planned): add graph-aware retrieval (relationships / graph expansion per Technical Plan), then wire the UI button. Enable once MiniRAG graph path is implemented.

---

# Phase 7 — Productionization

## Goals

- Docker
- GPU inference
- scaling
- observability

---

# 35. Architecture Phase 2 — Graph-aware upgrade (deferred)

**Status:** Deferred. Current delivery focuses on the Implementation Roadmap (Phases 1–7): markdown-first ingestion, enrichment, semantic chunk manifests, dense embeddings (BGE-M3), Qdrant, hybrid retrieval, reranking, and the in-repo `MiniRagService` orchestration. This section records the **optional second-wave architecture** for graph-backed indexing and topology-aware retrieval—**without** adopting the full [HKUDS/MiniRAG](https://github.com/HKUDS/MiniRAG) framework unless product requirements change.

## Purpose

- Improve **multi-hop** and **scattered-evidence** questions where seed retrieval (lexical and/or vector) hits only part of the answer.
- **Reuse sunk cost:** `*.chunks.json` (chunk ids, `relationships`, `continuation`) and `data/enriched/*.enrichment.json` (entities, soft relationships) become inputs to a materialized graph, not a parallel pipeline rewrite.

## Scope (when activated)

| Area | Direction |
|------|-----------|
| **Graph model** | Nodes: `document`, `chunk` (stable `chunk_id` / `document_id`), optional `entity`. Edges: `contains`, `continuation`, `see_also`, `prerequisite`, etc., aligned with existing chunk manifest and enrichment fields. |
| **Graph build** | Batch job after chunking (and optionally after enrichment updates): load manifests → upsert nodes/edges. Start with **SQLite** or **in-memory** adjacency; escalate to **Neo4j/Postgres** only if query complexity or ops require it. |
| **Retrieval** | **Seed** (keyword + dense from Qdrant) → **bounded expansion** (e.g. 1–2 hops on allowed edge types) → **dedupe + token budget** → optional **rerank** → existing **grounded prompt** + LM Studio. |
| **API / product** | Extend `MiniRagService` or add `GraphRagService`; expose mode flag on `/chat` if needed; enable **RagGraph Evaluation** in the UI once stable. |

## Explicit non-goals (Phase 2)

- Mandatory migration to HKUDS MiniRAG’s storage/runtime (e.g. forced LightRAG/Neo4j layout) without a dedicated integration decision.
- Replacing the **canonical** filesystem corpus under `backend/data/*`; the graph is an **index/projection**, not the source of truth.

## Dependencies

- Stable **chunk-level** identities in `*.chunks.json` (Step 7).
- **Chunk-level** vectors and payloads in Qdrant (Step 8) for strong seed retrieval before expansion.
- Evidence of **under-retrieval** on a labeled set of multi-hop queries (informs ROI).

## Success criteria

- Documented **node/edge schema** and idempotent **rebuild** procedure.
- **Retrieval traces** (seed chunks → expanded chunk ids) for debugging and evaluation.
- Measured improvement on a **small multi-hop benchmark** without unacceptable latency or noise on simple FAQ-style queries.

## Relationship to Implementation Roadmap

- **Phases 1–7** remain the primary schedule; **§35 here** is a **follow-on architecture track**, not a rename of roadmap “Phase 2 — Markdown Pipeline.”
- Section **17 (GraphRAG Concepts)** and **Evaluation modes → RagGraph** remain conceptual/UI placeholders until this deferred phase is executed.

---

# 36. Future Enhancements

## Potential Future Additions

- Neo4j graph database
- agentic workflows
- autonomous retrieval agents
- multimodal ingestion
- OCR support
- PDF ingestion
- user personalization
- feedback loops

---

# 37. Final Recommendations

## Most Important Success Factors

### 1. High-Quality Markdown

This is foundational.

---

### 2. Semantic Chunking

Better chunking improves retrieval more than larger LLMs.

---

### 3. Strong Metadata

Generate:

- tags
- entities
- questions answered
- relationships

---

### 4. Re-ranking

Never skip reranking.

---

### 5. Grounded Prompts

Prevent hallucinations.

---

## Final Recommended Architecture

```text
MiniRAG
+ Qwen3 8B Instruct
+ BGE-M3
+ bge-reranker-v2
+ Qdrant
+ Markdown-first ingestion
+ Semantic chunking
+ Relationship generation
```

This architecture is:

- production-ready
- scalable
- self-hosted
- GraphRAG-compatible
- enterprise-friendly
- cost-efficient
- optimized for technical documentation

