from fastapi import FastAPI
from pydantic import BaseModel
import re
import os
import requests
import uvicorn

# LangChain / RAG imports

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# ✅ Load heavy models ONCE at startup
# ─────────────────────────────────────────────

from langchain_huggingface import HuggingFaceEndpointEmbeddings

embed_model = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HF_API_KEY"),
    model="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatGroq(model="llama-3.1-8b-instant")

app = FastAPI()


# ─────────────────────────────────────────────
# ✅ SerpAPI Transcript Fetcher
# ─────────────────────────────────────────────
def get_transcript(video_id: str) -> str:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise Exception("SERPAPI_API_KEY environment variable is not set.")

    params = {
        "api_key": api_key,
        "engine": "youtube_video_transcript",
        "v": video_id,
        "type": "asr",  # Auto-generated captions (works for most videos)
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"SerpAPI request failed: {str(e)}")

    transcript_entries = data.get("transcript")

    if not transcript_entries:
        raise Exception(
            "No transcript found for this video. "
            "The video may not have auto-generated captions, or the video ID is invalid."
        )

    # Join all snippet texts into one transcript string
    full_transcript = " ".join(
        entry.get("snippet", "") for entry in transcript_entries
    )

    if not full_transcript.strip():
        raise Exception("Transcript was fetched but appears to be empty.")

    return full_transcript


# ─────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────
class Query(BaseModel):
    video_url: str
    question: str


# ─────────────────────────────────────────────
# Utility: extract video ID
# ─────────────────────────────────────────────
def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


# ─────────────────────────────────────────────
# MAIN API
# ─────────────────────────────────────────────
@app.post("/ask")
def ask_question(data: Query):
    try:
        # 1. Extract video ID
        video_id = extract_video_id(data.video_url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}

        # 2. Fetch transcript via SerpAPI
        try:
            transcript_text = get_transcript(video_id)
        except Exception as e:
            return {"error": str(e)}

        # 3. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript_text])

        # 4. Build vector store
        vector_store = FAISS.from_documents(chunks, embed_model)

        # 5. Retriever with MMR
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        # 7. Prompt
        prompt = PromptTemplate(
            template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
            input_variables=["context", "question"]
        )

        # 8. Format docs helper
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 9. Build and run chain
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        main_chain = parallel_chain | prompt | llm | StrOutputParser()
        response = main_chain.invoke(data.question)

        return {
            "answer": response,
            "video_id": video_id
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
