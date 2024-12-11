import time

from langsmith import evaluate

from backend.chat_models.llms import rag_model
from backend.evaluation.evaluator import RagEvaluator
from backend.vectorstore import MultiModalVectorstore

if __name__ == "__main__":
    evaluator = RagEvaluator(student_llm=rag_model, db=MultiModalVectorstore())

    evaluation_inputs = [
        (
            evaluator.predict_rag_answer,
            evaluator.answer_accuracy_evaluator,
            "answer-acc",
        ),
    ]
    for target, evaluator, description in evaluation_inputs:
        evaluate(
            target,
            data="OXFAM_dataset",
            evaluators=[evaluator],
            experiment_prefix=description,
            metadata={"version": "test"},
        )
