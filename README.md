# 🎓 Biyani AI Counselor - Premium RAG Chatbot

A high-intelligence, multi-provider RAG (Retrieval-Augmented Generation) chatbot designed for the Biyani Group of Colleges. It provides accurate information about admissions, courses, fees, and campus details using a smart model-switching architecture.

## 🚀 Key Features

- **Gold 6 Model Stack**: Intelligent switching between 4 providers (Groq, Gemini, NVIDIA, OpenRouter) for maximum reliability and intelligence.
- **Smart RAG Engine**: Advanced semantic search using Qdrant Vector Database and `BGE-Small` embeddings.
- **Natural Personality**: Professional yet warm Academic Counselor persona that bridges information gaps intelligently.
- **Language Intelligence**: Seamlessly detects and responds in English or natural Hinglish based on the user's input.
- **Smart Model Selector**: Automatically chooses between 70B+ models for complex queries and faster 8B models for simple ones.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **Vector DB**: Qdrant Cloud
- **Embeddings**: BAAI/bge-small-en-v1.5 (Local)
- **LLM Providers**:
  - **GROQ**: Llama 3.3 70B & 3.1 8B (Speed)
  - **GEMINI**: 2.5 Flash & Flash Lite (Reasoning)
  - **NVIDIA**: Mistral Large 3 (675B) & Dracarys 70B (Power)
  - **OPENROUTER**: Nemotron 120B (Stability)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kushal96499/Biyani-AI-Counselor.git
   cd Biyani-AI-Counselor
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file and add your keys:
   ```env
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_key
   GROQ_API_KEY=your_key
   GEMINI_API_KEY=your_key
   NVIDIA_API_KEY=your_key
   OPENROUTER_API_KEY=your_key
   ```

## 🌐 Deployment (Vercel)

This project is configured for Vercel Serverless Functions.

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel login` and `vercel` to deploy.
3. Ensure all Environment Variables are added in the Vercel Dashboard.

---
Built with ❤️ for Biyani Group of Colleges.
