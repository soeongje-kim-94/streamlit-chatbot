import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import answer_examples
from factory import LLMFactory, VectorStoreFactory
from history import format_history, get_history
from vector import format_documents

load_dotenv()

llm = LLMFactory().get_instance()
database = VectorStoreFactory(os.environ.get("PINECONE_API_KEY")).get_instance()


def get_dictionary_chain():
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
        
        [사전]
        {dictionary}
        
        [현재 질문]
        {question}
        """
    ).partial(dictionary="\n".join(dictionary))

    return dictionary_prompt | llm | StrOutputParser()


def get_history_chain():
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


def get_summary_chain():
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신의 역할은 대화 이력을 간결하게 요약하는 것입니다.",
            ),
            (
                "human",
                """
                아래는 지금까지의 대화 이력입니다.
                이 대화를 간결하게 요약해주세요.
                
                [대화 이력]
                {history}
                
                [요약]
                """,
            ),
        ]
    )

    return summary_prompt | llm | StrOutputParser()


def get_initial_question_chain():
    initial_question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 세무 상담 서비스의 질문 추천 엔진입니다.
                사용자의 관심사를 기반으로, 실제 사용자가 많이 물어볼 법한 명확하고 구체적인 질문을 추천해야 합니다.
                """,
            ),
            (
                "human",
                """
                아래 관심사에 맞는 질문 {k}개를 추천해주세요.
                질문은 짧고 명확해야 하며, 실제 사용자가 입력할 수 있는 형태여야 합니다.
                
                [관심사]
                {interests}
                """,
            ),
        ]
    )

    return initial_question_prompt | llm | StrOutputParser()


def get_question_chain():
    question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 문서를 기반으로 사용자 질문을 생성하는 전문가입니다.
                질문은 실제 사용자가 궁금해할 법한 형태여야 합니다.
                """,
            ),
            (
                "human",
                """
                다음 문서를 보고 사용자가 물어볼 만한 질문 3~5개를 생성해주세요.

                [문서]
                {document}
                """,
            ),
        ]
    )

    return question_prompt | llm | StrOutputParser()


def get_rag_chain(session_id):
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
            ("ai", "{answer}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 소득세법 전문가입니다.
                소득세법에 관한 사용자의 질문에 답변해주세요.
                """,
            ),
            few_shot_prompt,
            (
                "system",
                """
                아래는 이전 대화를 요약한 내용입니다.
                답변 시 이 맥락을 참고하되, 불필요하게 반복하지는 마세요.
            
                [대화 요약]
                {summary}
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

    history = get_history(session_id)
    retriever = database.as_retriever()

    history_chain = get_history_chain()
    summary_chain = get_summary_chain()

    return (
        {
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: format_history(history)),
        }
        | RunnableLambda(
            lambda x: {
                "question": x["question"],
                "history": x["history"],
                "standalone_question": history_chain.invoke(
                    {
                        "question": x["question"],
                        "history": x["history"],
                    }
                ),
            }
        )
        | RunnableLambda(
            lambda x: {
                **x,
                "summary": summary_chain.invoke({"history": x["history"]}),
            }
        )
        | {
            "question": lambda x: x["question"],
            "summary": lambda x: x["summary"],
            "docs": lambda x: retriever.invoke(x["standalone_question"]),
        }
        | {
            "question": lambda x: x["question"],
            "summary": lambda x: x["summary"],
            "context": lambda x: format_documents(x["docs"]),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
