# YouTube Chatbot — Chrome Extension with RAG Backend

A Chrome Extension that lets you ask questions about any YouTube video in real time. It extracts the video transcript via SerpAPI, builds a FAISS vector index on the fly, and answers your questions using a LangChain RAG pipeline powered by Groq LLM and Cohere Embeddings — all through a clean, YouTube-styled popup UI.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/RAG-LangChain-blue)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://groq.com/)
[![Cohere](https://img.shields.io/badge/Embeddings-Cohere-purple)](https://cohere.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-blue)](https://faiss.ai/)
[![Chrome Extension](https://img.shields.io/badge/Extension-Manifest%20V3-yellow?logo=googlechrome)](https://developer.chrome.com/docs/extensions/mv3/)

---

## Problem Statement

Watching a long YouTube video just to find one specific piece of information is time-consuming. This project solves that by letting you chat directly with any YouTube video — ask it questions, get instant answers — without watching a single second more than you need to.

---

## 🚀 Live Demo(Linkedin post with Live working Video)

-> [Click here to view the live working & projects url](https://www.linkedin.com/feed/update/urn:li:activity:7457797307330084864/)

---

## Architecture

```
User (Chrome Extension Popup)
        │
        │  POST /ask  { video_url, question }
        ▼
  FastAPI Backend  (Render)
        │
        ├── SerpAPI  →  Fetch video transcript
        ├── RecursiveCharacterTextSplitter  →  Chunk transcript
        ├── Cohere Embeddings  →  embed-english-v3.0
        ├── FAISS Vector Store  →  MMR Retrieval (k=5)
        └── Groq LLM  →  llama-3.1-8b-instant
                │
                ▼
         RAG Chain  →  Answer streamed back to popup
```

---

## Features

| Feature | Detail |
|---------|--------|
|  **Transcript Extraction** | Fetches auto-generated captions via SerpAPI for any YouTube video |
|  **RAG Pipeline** | Per-request FAISS index built from chunked transcript |
|  **MMR Retrieval** | Maximal Marginal Relevance for diverse, high-quality context chunks |
|  **Fast LLM** | Groq's `llama-3.1-8b-instant` for near-instant answers |
|  **Chrome Extension** | YouTube-themed popup UI with chat interface, Manifest V3 |
|  **Render Deployment** | Backend hosted on Render with dynamic port support |

---

## Project Structure

```
youtube-chatbot/
├── app.py                  # FastAPI backend — RAG pipeline, SerpAPI transcript fetch
├── requirements.txt        # Python dependencies
├── cookies.txt             # YouTube cookies (for restricted video access)
├── .env                    # API keys (not committed)
│
└── extension/              # Chrome Extension files
    ├── manifest.json       # Extension config (Manifest V3)
    ├── popup.html          # Extension popup UI
    ├── popup.js            # Popup logic — chat, fetch, rendering
    ├── content.js          # Content script — reads active YouTube tab URL
    └── icons/              # Extension icons (16, 48, 128px)
```

---

## Quickstart

### 1. Clone & Set Up Backend

```bash
git clone https://github.com/your-username/youtube-chatbot-chromeplugin.git
cd youtube-chatbot-chromeplugin

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

### 3. Run the Backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Load the Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **Load unpacked** and select the `extension/` folder
4. Navigate to any YouTube video and click the extension icon

---

## API

### `POST /ask`

**Request Body:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "question": "What is this video about?"
}
```

**Response:**
```json
{
  "answer": "This video is about...",
  "video_id": "VIDEO_ID"
}
```

**Error Response:**
```json
{
  "error": "No transcript found for this video."
}
```

---

## How It Works

### RAG Pipeline (Per Request)

Each `/ask` request triggers a fresh pipeline:

1. **Transcript Fetch** — SerpAPI fetches auto-generated captions for the given video ID using the `youtube_video_transcript` engine.
2. **Chunking** — The full transcript is split into 1000-character chunks (200-character overlap) using `RecursiveCharacterTextSplitter`.
3. **Embedding** — Chunks are embedded with Cohere's `embed-english-v3.0` model.
4. **Vector Store** — A FAISS index is built in-memory from the embedded chunks.
5. **MMR Retrieval** — The retriever fetches the top 5 most relevant and diverse chunks (`fetch_k=20`, `lambda_mult=0.7`).
6. **LLM Answer** — Retrieved chunks are passed as context to Groq's `llama-3.1-8b-instant` with a strict "answer only from transcript" prompt.

### Chrome Extension Flow

The extension uses Manifest V3 with a content script injected on all `youtube.com/*` pages. When the popup opens, it reads the active tab URL and — if it's a YouTube video — renders the chat interface. Questions are sent directly to the hosted FastAPI backend, and responses are displayed in the chat bubble UI.

---

## Deployment (Render)

The backend is deployed on [Render](https://render.com) as a Web Service. The port is read dynamically from the `PORT` environment variable:

```python
port = int(os.environ.get("PORT", 10000))
uvicorn.run("app:app", host="0.0.0.0", port=port)
```

The extension's `popup.js` points to the live Render URL:

```javascript
const BACKEND_URL = "https://youtube-chatbot-chromeplugin.onrender.com/ask";
```

To redeploy with your own backend, update this URL in `popup.js` and reload the extension.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for the backend API |
| `uvicorn` | ASGI server |
| `langchain` | RAG chain orchestration |
| `langchain-groq` | Groq LLM integration |
| `langchain-community` | FAISS vector store, document loaders |
| `langchain-cohere` | Cohere embedding integration |
| `faiss-cpu` | Vector similarity search |
| `requests` | SerpAPI HTTP calls |
| `python-dotenv` | Environment variable management |

---

## Roadmap

- [ ] Cache transcripts to avoid re-fetching the same video on repeated questions
- [ ] Support multilingual transcripts (Hindi, Spanish, etc.)
- [ ] Add conversation history for multi-turn Q&A
- [ ] Timestamp-aware answers — link responses back to video timestamps
- [ ] Support YouTube Shorts

