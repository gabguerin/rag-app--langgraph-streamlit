from langsmith import evaluate

from backend.chat_models.llms import rag_model
from backend.evaluation.evaluator import RagEvaluator
from backend.vectorstore import MultiModalVectorstore

if __name__=="__main__":
    evaluator = RagEvaluator(student_llm=rag_model, db=MultiModalVectorstore())

    evaluation_inputs = [
        (evaluator.predict_rag_answer_with_context, evaluator.document_relevancy_evaluator),
        (evaluator.predict_rag_answer_with_context, evaluator.answer_hallucination_evaluator),
        (evaluator.predict_rag_answer, evaluator.answer_accuracy_evaluator),
        (evaluator.predict_rag_answer, evaluator.answer_helpfulness_evaluator),
    ]
    for target, evaluator in evaluation_inputs:
        evaluate(
            target,
            data=dataset_name,
            evaluators=[evaluator],
            experiment_prefix="rag-doc-relevance",
            metadata={"version": "test"},
        )
