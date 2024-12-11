from langsmith import evaluate

from backend.chat_models.llms import rag_model
from backend.evaluation.evaluator import RagEvaluator
from backend.vectorstore import MultiModalVectorstore

if __name__ == "__main__":
    rag_evaluator = RagEvaluator(student_llm=rag_model, db=MultiModalVectorstore())

    evaluation_inputs = [
        (
            rag_evaluator.answer_accuracy_evaluator,
            "answer-acc",
        ),
    ]
    for metric_evaluator, description in evaluation_inputs:
        evaluate(
            rag_evaluator.run,
            data="OXFAM_dataset",
            evaluators=[metric_evaluator],
            experiment_prefix=description,
            metadata={"version": "test"},
        )
