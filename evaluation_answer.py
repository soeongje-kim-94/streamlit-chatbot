from langsmith.evaluation import evaluate

from evaluation import predict_rag_answer
from evaluation import answer_evaluator, answer_helpfulness_evaluator
from dataset import dataset

experiment_results = evaluate(
    predict_rag_answer,
    data=dataset.name,
    evaluators=[answer_evaluator, answer_helpfulness_evaluator],
    experiment_prefix="streamlit-income-tax-answer",
    metadata={"version": "income tax v1, gpt-4o-mini"},
)
