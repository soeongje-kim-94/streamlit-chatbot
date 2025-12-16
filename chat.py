import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langsmith import Client
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone
from operator import itemgetter

st.set_page_config(page_title="소득세 챗봇", page_icon="💬")
st.title("💬 소득세 챗봇")
st.caption("소득세 관련 질문에 답변해 드립니다.")

load_dotenv()

# Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# LangSmith
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
langsmith_client = Client(api_key=langsmith_api_key)

# Embeddings & Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "tax-index-large"
database = PineconeVectorStore(embedding=embeddings, index=pc.Index(index_name))

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

def get_ai_message(user_input):
    dictionary = [
        "사람을 나타내는 표현(예: 사람, 인간, 개인, 시민, 주민, 납세자) → 거주자",
    ]

    dictionary_prompt = ChatPromptTemplate.from_template(
        """
        사용자의 질문을 보고, 야래 제공된 사전을 참고하여 질문을 변경해주세요.
        만일 변경할 필요가 없다고 판단되면, 질문을 변경하지 않고 그대로 반환해주세요.
        
        # 주의사항
        - 설명, 접두어, 접미어, 따옴표를 절대 포함하지 말것
        - "변경된 질문", "수정된 질문" 등과 같은 문구를 포함하지 말것
        
        사전:
        {dictionary}
        
        질문:
        {question}
        """
    ).partial(dictionary="\n".join(dictionary))
    dictionary_chain = (
        dictionary_prompt 
        | llm 
        | StrOutputParser()
    )

    qa_prompt = langsmith_client.pull_prompt("teddynote/rag-prompt-korean", include_model=True)
    qa_chain = (
        dictionary_chain
        | {
            "context": database.as_retriever(search_kwargs={"k": 4}),
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain.invoke(user_input)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input(placeholder="소득세에 대해 무엇이든 물어보세요!"):
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("답변을 생성하는 중입니다..."):
        ai_message = get_ai_message(user_input)

        with st.chat_message("ai"):
            st.write(ai_message)

        st.session_state.messages.append({"role": "ai", "content": ai_message})
