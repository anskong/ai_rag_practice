
import streamlit as st
import pandas as pd


import os
from dotenv import load_dotenv

# 환경변수 읽어오기
load_dotenv(override=True)  # .env 파일을 덮어쓰기 모드로 읽기

# 환경변수 불러오기
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 환경변수 불러오기
# openai_key = os.getenv("OPENAI_API_KEY")
# anthropic_key = os.getenv("ANTHROPIC_API_KEY")
# huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# print(f"openai key values ::: {openai_key}")  # 테스트용 (실제 서비스에서는 print 금지)
# print(f"anthropic key values ::: {anthropic_key}")  # 테스트용 (실제 서비스에서는 print 금지)
# print(f"huggingface_token::: {huggingface_token}")  # 테스트용 (실제 서비스에서는 print 금지)

st.title("chatBot")
st.write("Chatbot with Anthropic Claude 3")

#session_state에 messages Key값 지정 및 Streamlit 화면 진입 시, AI의 인사말을 기록하기
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

#사용자나 AI가 질문/답변을 주고받을 시, 이를 기록하는 session_state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#챗봇으로 활용할 AI 모델 선언
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
import httpx  

vfy_client = httpx.Client(verify=False)

# 1. 직접 Anthropic
client = Anthropic(api_key=anthropic_key, http_client=vfy_client)

# 2. Langchain Anthropic 모델 호출
chat = ChatAnthropic(
    model_name ="claude-3-opus-20240229",
    anthropic_api_key=anthropic_key,)

chat._client = client

#chat_input()에 입력값이 있는 경우,
if prompt := st.chat_input():
    #messages라는 session_state에 역할은 사용자, 컨텐츠는 프롬프트를 각각 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    #chat_message()함수로 사용자 채팅 버블에 prompt 메시지를 기록
    st.chat_message("user").write(prompt)

    
    response = chat.invoke(prompt)
    msg = response.content

    #messages라는 session_state에 역할은 AI, 컨텐츠는 API답변을 각각 저장
    st.session_state.messages.append({"role": "assistant", "content": msg})
    #chat_message()함수로 AI 채팅 버블에 API 답변을 기록
    st.chat_message("assistant").write(msg)


# st.write(st.session_state.messages.len)
st.write(len(st.session_state.messages))
# st.write(st.session_state.messages)


