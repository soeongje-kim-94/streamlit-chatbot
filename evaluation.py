from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from dotenv import load_dotenv

from llm import create_llm, get_retriever

load_dotenv()


class RagBot:

    def __init__(self, retriever, model: str = "gpt-4o-mini"):
        self._client = wrap_openai(OpenAI())
        self._retriever = retriever
        self._model = model

    @traceable()
    def retrieve_docs(self, question):
        return self._retriever.invoke(question)

    @traceable()
    def invoke_llm(self, question, docs):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    당신은 한국의 소득세 전문가입니다.
                    아래 소득세법을 참고해서 사용자의 질문에 답변해주세요.
                    
                    [소득세법]
                    {docs}
                    """,
                },
                {"role": "user", "content": question},
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in docs],
        }

    @traceable()
    def get_answer(self, question: str):
        docs = self.retrieve_docs(question)

        return self.invoke_llm(question, docs)


llm = create_llm()
retriever = get_retriever()
rag_bot = RagBot(retriever)
langsmith_client = Client()


def predict_rag_answer(example: dict):
    """답변만 평가할 때 사용"""
    response = rag_bot.get_answer(example["input_question"])

    return {"answer": response["answer"]}


def predict_rag_answer_with_context(example: dict):
    """Context를 활용해서 hallucination을 평가할 때 사용"""
    response = rag_bot.get_answer(example["input_question"])

    return {"answer": response["answer"], "contexts": response["contexts"]}


def answer_evaluator(run, example) -> dict:
    """
    RAG 답변 성능을 측정하기 위한 evaluator
    """

    grade_prompt_answer_accuracy = langsmith_client.pull_prompt("langchain-ai/rag-answer-vs-reference")
    answer_grader = grade_prompt_answer_accuracy | llm

    score = answer_grader.invoke(
        {
            "question": example.inputs["question"],         # input
            "correct_answer": example.outputs["answer"],    # reference
            "student_answer": run.outputs["answer"],        # prediction
        }
    )
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}


def answer_helpfulness_evaluator(run, example) -> dict:
    """
    답변이 사용자의 질문에 얼마나 도움되는지 판단하는 evaluator
    """

    grade_prompt_answer_helpfulness = langsmith_client.pull_prompt("langchain-ai/rag-answer-helpfulness")
    answer_grader = grade_prompt_answer_helpfulness | llm

    score = answer_grader.invoke(
        {
            "question": example.inputs["question"],     # input
            "student_answer": run.outputs["answer"]     # prediction
        }
    )
    score = score["Score"]

    return {"key": "answer_helpfulness_score", "score": score}


def answer_hallucination_evaluator(run, example) -> dict:
    """
    Hallucination 판단을 위한 evaluator
    """

    grade_prompt_hallucinations = langsmith_client.pull_prompt("langchain-ai/rag-answer-hallucination")
    answer_grader = grade_prompt_hallucinations | llm

    score = answer_grader.invoke(
        {
            "documents": example.inputs["contexts"],    # input
            "student_answer": run.outputs["answer"]     # prediction
        }
    )
    score = score["Score"]

    return {"key": "answer_hallucination", "score": score}