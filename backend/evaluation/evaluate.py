from langsmith import evaluate

from backend.chat_models.llms import rag_model
from backend.evaluation.evaluator import RagEvaluator
from backend.vectorstore import MultiModalVectorstore

if __name__ == "__main__":
    rag_evaluator = RagEvaluator(student_llm=rag_model, db=MultiModalVectorstore())

    evaluation_inputs = {
        "answer_accuracy": rag_evaluator.answer_accuracy_evaluator,
    }
    for metric_id, metric_evaluator in evaluation_inputs.items():
        evaluate(
            rag_evaluator.run,
            data="OXFAM_dataset",
            evaluators=[metric_evaluator],
            experiment_prefix=metric_id,
            metadata={"version": "test"},
        )
