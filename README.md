# 🎓 Biyani AI Counselor - Ultra-Premium RAG Architecture

![React](https://img.shields.io/badge/React-18.0-blue?style=flat-square&logo=react)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC?style=flat-square&logo=tailwind-css)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square&logo=fastapi)
![Redis](https://img.shields.io/badge/Cache-Upstash--Redis-orange?style=flat-square&logo=redis)
![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-black?style=flat-square&logo=vercel)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red?style=flat-square&logo=database)

A state-of-the-art, high-intelligence **Retrieval-Augmented Generation (RAG)** chatbot engineered specifically for the **Biyani Group of Colleges**. 

This system is built with a **100% Serverless Architecture**, designed to bypass traditional deployment constraints by utilizing cloud-based embedding APIs, a high-speed Redis caching layer, and a non-blocking asynchronous backend.

---

## 🏗️ Technical Architecture

The pipeline is completely cloud-native and asynchronous. Every request is handled concurrently to support multiple users simultaneously.

```mermaid
graph TD
    User([User/Student]) -->|Async Request| Web[Frontend: Zero-Build React + Tailwind]
    Web -->|POST /api/chat| Backend[FastAPI Async Server]
    
    subgraph "Intelligent Cache Layer"
        Backend -->|Check Cache| Redis{Upstash Redis}
        Redis -- Exact/Semantic Hit --> User
    end

    subgraph "Cloud RAG Engine"
        Redis -- Miss --> Embedder[Cloud Embeddings API]
        Embedder -->|Vector Search| Qdrant[(Qdrant Cloud DB)]
        Qdrant -->|Context Retrieval| Reranker[Reranker: rerank-qa-mistral-4b]
        Reranker -->|Top-K Context| Engine{Smart RAG Engine}
    end

    subgraph "Tiered LLM Intelligence Stack"
        Engine -->|Priority 1| Groq[Groq: Llama 3.3 70B]
        Engine -->|Priority 2| NVIDIA[NVIDIA NIM: Solar/Mistral]
        Engine -->|Fallback| OR[OpenRouter: Nemotron 120B]
    end

    Groq & NVIDIA & OR -->|AI Response| CacheStore[Save to Redis]
    CacheStore -->|Regex Formatted CTA| User
```

---

## 🌟 Premium Features

### 1. High-Speed Redis Caching (Exact + Semantic)
- **Sub-10ms Responses**: Uses Upstash Redis (REST) for instant exact-match query retrieval.
- **Semantic Reusability**: Uses vector similarity (0.88 threshold) to reuse answers for similar phrasing, saving API costs and improving speed.
- **Dynamic TTL**: Critical data (fees/admissions) expires faster than static info (about college).

### 2. Async Multi-User Concurrency
- **Non-blocking Backend**: Built on **FastAPI Async**, allowing the server to handle hundreds of requests without sequential blocking.
- **Queue Tracking**: Real-time tracking of active requests to inform users of the system status during peak traffic.

### 3. Ultra-Premium UI/UX
- **Standalone React Architecture**: Runs directly in `index.html` via CDN for zero-build deployment.
- **Glassmorphism Design**: Apple-like `backdrop-blur-3xl` and multi-layered glowing background blobs.
- **Frame-Independent Typing**: Uses `requestAnimationFrame` to ensure typing effects never pause when the tab is in the background.

### 4. Vercel-Optimized Performance
- **Lightweight Payload**: Completely removed heavy local ML models (fastembed/pypdf) to fit Vercel's 250MB limit.
- **Tiered Cloud Failover**: 100% crash-proof logic. If one provider rate-limits, it instantly fails over to the next in <200ms.

---

## 🛠️ Technology Stack

### Backend & AI
- **Framework**: FastAPI (Asynchronous Python 3.12)
- **Vector Database**: Qdrant Cloud
- **Cache**: Upstash Redis (REST API)
- **Reranker Engine**: Mistral QA Reranker (`rerank-qa-mistral-4b`)
- **LLM Engine**:
  - **GROQ**: Llama 3.3 70B / 3.1 8B *(Primary)*
  - **NVIDIA NIM**: Upstage Solar / Llama 3.1 70B *(Secondary)*
  - **OPENROUTER**: Nemotron / Gemma *(Fallback)*

### Frontend
- **Framework**: React 18 (CDN Standalone)
- **Styling**: Tailwind CSS 3.0 (CDN)
- **Typography**: Google Fonts (Outfit / Inter)

---

## 📦 Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kushal96499/Biyani-AI-Counselor.git
   cd Biyani-AI-Counselor
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory and add your keys:
   ```env
   # Vector DB & Cache
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_key
   UPSTASH_REDIS_REST_URL=your_url
   UPSTASH_REDIS_REST_TOKEN=your_token

   # LLM Providers
   GROQ_API_KEY=your_key
   NVIDIA_API_KEY=your_key
   OPENROUTER_API_KEY=your_key
   ```

3. **Install & Run**:
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

---

## 👨‍💻 Internship & Development
This project was developed as part of an official **College Internship** at the **Biyani Web Cell**.

- **Developer**: **Kushal Kumawat** (BCA 3rd Year Student)
- **Institution**: Biyani Institute of Science & Management, Jaipur
- **Internship Mentor**: Developed under the esteemed guidance of **Pankaj Sir** (Web Cell Head).
- **Project Scope**: Research and implementation of AI-driven counseling automation for the Biyani Group of Colleges.

---
*Built with ❤️ by **Kushal Kumawat** to empower students at the Biyani Group of Colleges.*
