import streamlit as st
from dotenv import load_dotenv

from llm import add_ai_message, get_ai_response

st.set_page_config(page_title="소득세 챗봇", page_icon="💬")
st.title("💬 소득세 챗봇")
st.caption("소득세 관련 질문에 답변해 드립니다.")

load_dotenv()

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
        ai_response = get_ai_response(user_input)

        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.messages.append({"role": "ai", "content": ai_message})

            add_ai_message(ai_message)
