import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langsmith import Client
from pinecone import Pinecone

# Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = "tax-index-large"

# LangSmith
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")

store = {}

load_dotenv()


def create_vector_store(index_name, api_key):
    pc = Pinecone(api_key=api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return PineconeVectorStore(
        embedding=embeddings,
        index=pc.Index(index_name),
    )


def create_llm(model_name="gpt-4o-mini"):
    return ChatOpenAI(model=model_name)


def create_langsmith_client(api_key):
    return Client(api_key=api_key)


def get_retriever():
    database = create_vector_store(pinecone_index_name, pinecone_api_key)

    return database.as_retriever(search_kwargs={"k": 4})


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


def format_documents(documents):
    return "\n\n".join(d.page_content for d in documents)


def format_history(history: InMemoryChatMessageHistory):
    return "\n".join(f"{m.type.upper()}: {m.content}" for m in history.messages)


def get_dictionary_chain():
    dictionary = [
        "사람을 나타내는 표현(예: 사람, 인간, 개인, 시민, 주민, 납세자) → 거주자",
    ]

    llm = create_llm()

    dictionary_prompt = ChatPromptTemplate.from_template(
        """
        사용자의 질문을 보고, 야래 제공된 사전을 참고하여 질문을 변경해주세요.
        만일 변경할 필요가 없다고 판단되면, 질문을 변경하지 않고 그대로 반환해주세요.
        
        # 주의사항
        - 설명, 접두어, 접미어, 따옴표를 절대 포함하지 말것
        - "변경된 질문", "수정된 질문" 등과 같은 문구를 포함하지 말것
        
        [사전]
        {dictionary}
        
        [현재 질문]
        {question}
        """
    ).partial(dictionary="\n".join(dictionary))

    return dictionary_prompt | llm | StrOutputParser()


def get_conversion_chain():
    llm = create_llm()

    conversion_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신의 역할은 사용자의 질문을 문서 검색에 적합한 하나의 독립된 질문으로 재작성하는 것입니다.",
            ),
            (
                "human",
                """
                아래는 지금까지의 대화 이력입니다.
                이 대화를 참고하여, 현재 사용자의 질문을 이전 맥락 없이도 이해할 수 있는 '검색용 독립 질문'으로 다시 작성해주세요.
                
                [대화 이력]
                {history}
                
                [질문]
                {question}
                
                [검색용 독립 질문]
                """,
            ),
        ]
    )

    return conversion_prompt | llm | StrOutputParser()


def get_rag_chain(session_id):
    llm = create_llm()
    history = get_history(session_id)
    retriever = get_retriever()

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 소득세법 전문가입니다.
                소득세법에 관한 사용자의 질문에 답변해주세요.
                """,
            ),
            (
                "system",
                """
                아래는 이전 대화를 요약한 내용입니다.
                답변 시 이 맥락을 참고하되, 불필요하게 반복하지는 마세요.
            
                [대화 이력]
                {history}
                """,
            ),
            (
                "system",
                """
                아래에 제공된 문서를 활용해서 답변해주시고, 답변을 알 수 없다면 모른다고 답변해주세요.
                이 정보에 포함되지 않은 내용은 추측하거나 만들어내지 마세요.
                
                [검색 문서]
                {context}
                """,
            ),
            (
                "human",
                """
                사용자의 질문에 대해 위의 정보만들 사용하여, 명확하고 간결하게 답변하세요.
                답변을 제공할 때는 '소득세법 (XX조)에 따르면'이라고 시작하면서 답변해주시고,
                2~3 문장정도의 짧은 내용의 답변을 원합니다.
                
                [질문]
                {question}
                """,
            ),
        ]
    )

    conversation_chain = get_conversion_chain()

    return (
        {
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: format_history(history)),
        }
        | RunnableLambda(
            lambda x: {
                "question": x["question"],
                "history": x["history"],
                "standalone_question": conversation_chain.invoke(
                    {
                        "question": x,
                        "history": format_history(history),
                    }
                ),
            }
        )
        | {
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
            "docs": lambda x: retriever.invoke(x["standalone_question"]),
        }
        | {
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
            "context": lambda x: format_documents(x["docs"]),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )


def add_user_message(message, session_id="default"):
    history = get_history(session_id)
    history.add_user_message(message)

    return message


def add_ai_message(message, session_id="default"):
    history = get_history(session_id)
    history.add_ai_message(message)

    return message


def get_ai_response(user_input, session_id="default"):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain(session_id)
    tax_chain = (
        dictionary_chain
        | RunnableLambda(lambda x: add_user_message(x, session_id))
        | rag_chain
    )

    return tax_chain.stream(
        user_input, config={"configurable": {"session_id": session_id}}
    )
