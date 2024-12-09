from langchain import hub
from langsmith import evaluate

from backend.chat_models.llms import rag_model
from backend.evaluator.evaluator import RagEvaluator
from backend.vectorstore import MultiModalVectorstore

evaluator = RagEvaluator(student_llm=rag_model, db=MultiModalVectorstore())


def answer_accuracy_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """

    # Run evaluator
    score = evaluator.evaluate(
        metric=hub.pull("langchain-ai/rag-answer-vs-reference"),
        inputs={
            "question": example.inputs["question"],
            "correct_answer": example.outputs["output_answer"],
            "student_answer": run.outputs["answer"],
        },
    )

    return {"key": "answer_v_reference_score", "score": score}
