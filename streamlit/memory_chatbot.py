import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# 환경변수 읽어오기
load_dotenv(override=True)  # .env 파일을 덮어쓰기 모드로 읽기

# 환경변수 불러오기
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#Streamlit에서는 @st.cache_resource를 통해 한번 실행한 자원을 리로드 시에 재실행하지 않도록 캐시메모리에 저장할 수 있습니다.
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Create a vector store from the document chunks
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-small'))
    return vectorstore

# Initialize the LangChain components
@st.cache_resource
def initialize_components(selected_model):
    file_path = r"/Users/youngho_moon/Library/CloudStorage/GoogleDrive-anskong@gmail.com/내 드라이브/LLM-RAG-LangChain/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = create_vector_store(pages)
    retriever = vectorstore.as_retriever()

    # Define the contextualize question prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # Define the answer question prompt
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
