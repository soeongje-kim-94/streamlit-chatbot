from langsmith.evaluation import evaluate

from evaluation import predict_rag_answer_with_context
from evaluation import answer_hallucination_evaluator
from dataset import dataset

experiment_results = evaluate(
    predict_rag_answer_with_context,
    data=dataset.name,
    evaluators=[answer_hallucination_evaluator],
    experiment_prefix="streamlit-income-tax-hallucination",
    metadata={"version": "income tax v1, gpt-4o-mini"},
)
