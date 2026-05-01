from fastapi import FastAPI
from pydantic import BaseModel
import re
import uvicorn
import os
import requests

# LangChain / RAG imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    IpBlocked
)

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Request schema
class Query(BaseModel):
    video_url: str
    question: str


# 🔧 Utility: extract video ID
def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


# 🚀 MAIN API
@app.post("/ask")
def ask_question(data: Query):
    try:
        video_id = extract_video_id(data.video_url)

        if not video_id:
            return {"error": "Invalid YouTube URL"}

        # 📺 Fetch transcript
        ytt_api = YouTubeTranscriptApi()
        try:
            transcript = ytt_api.list(video_id)
            transcript_data = transcript.find_transcript(["en", "hi"]).fetch()
        except (TranscriptsDisabled, NoTranscriptFound, IpBlocked):
            return {"error": "Transcript not available for this video"}

        transcript_text = " ".join([chunk.text for chunk in transcript_data])

        # ✂️ Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript_text])

        # 🔗 Embeddings + Vector DB
        embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
       )

        vector_store = FAISS.from_documents(chunks, embed_model)

        # 🤖 LLM
        model = ChatGroq(model="llama-3.1-8b-instant")

        # 🔍 Retriever
        base_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        compressor = LLMChainExtractor.from_llm(model)

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # 📄 Prompt
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
            input_variables=["context", "question"]
        )

        # 🧠 Formatting
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 🔗 Chain
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        parser = StrOutputParser()

        main_chain = parallel_chain | prompt | model | parser

        # 💬 Get answer
        response = main_chain.invoke(data.question)

        return {
            "answer": response,
            "video_id": video_id
        }

    except Exception as e:
        return {"error": str(e)}
    