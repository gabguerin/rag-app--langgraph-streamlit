from typing import List, Any

from langchain import hub
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langsmith import traceable

from backend.chat_models.llms import LLM
from backend.utils import format_documents
from backend.vectorstore import MultiModalVectorstore


class RagEvaluator:
    """
    A class to evaluate Retrieval-Augmented Generation (RAG) workflows.
    It provides methods to retrieve documents, generate answers, and evaluate
    the quality of the RAG process using predefined metrics.
    """

    def __init__(self, student_llm: LLM, db: MultiModalVectorstore):
        """
        Initialize the RAG evaluator.

        Args:
            student_llm (LLM): The LLM used by the student model.
            db (MultiModalVectorstore): The vector store for document retrieval.
        """
        self._student_llm = student_llm
        self._db = db
        self._evaluator_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    @traceable()
    def retrieve_docs(self, question: str) -> List[Document]:
        """
        Retrieve relevant documents from the vector store based on the question.

        Args:
            question (str): The question for which documents are to be retrieved.

        Returns:
            List[Document]: A list of retrieved documents.
        """
        return self._db.retrieve_documents(question)

    @traceable()
    def invoke_llm(self, question: str, documents: List[Document]) -> dict[str, Any]:
        """
        Invoke the student LLM to generate an answer using the provided context.

        Args:
            question (str): The question to answer.
            documents (List[Document]): Context documents to guide the answer generation.

        Returns:
            dict[str, Any]: Contains the generated answer and the context.
        """
        response = self._student_llm.invoke(
            inputs={"context": format_documents(documents), "question": question}
        )

        return {
            "answer": response,
            "contexts": [doc.page_content for doc in documents],
        }

    @traceable()
    def get_answer(self, question: str) -> dict[str, Any]:
        """
        Retrieve documents and generate an answer for the question.

        Args:
            question (str): The question to answer.

        Returns:
            dict[str, Any]: The generated answer and associated contexts.
        """
        documents = self.retrieve_docs(question)
        return self.invoke_llm(question, documents)

    def predict_rag_answer(self, example: dict) -> dict[str, Any]:
        """
        Generate an answer without returning the associated context.

        Args:
            example (dict): Input example containing the question.

        Returns:
            dict[str, Any]: The generated answer.
        """
        response = self.get_answer(example["question"])
        return {"answer": response["answer"]}

    def predict_rag_answer_with_context(self, example: dict) -> dict[str, Any]:
        """
        Generate an answer and include the retrieved context.

        Args:
            example (dict): Input example containing the question.

        Returns:
            dict[str, Any]: The generated answer and associated contexts.
        """
        response = self.get_answer(example["question"])
        return {"answer": response["answer"], "contexts": response["contexts"]}

    def evaluate(self, metric: Any, inputs: dict[str, str]) -> Any:
        """
        Evaluate the generated answer using the specified metric.

        Args:
            metric (Any): The evaluation metric.
            inputs (dict[str, str]): Inputs required by the metric.

        Returns:
            Any: The evaluation score.
        """
        evaluator = metric | self._evaluator_llm  # Chain metric with evaluator LLM.
        score = evaluator.invoke(input=inputs)
        return score["Score"]

    # Individual evaluation methods
    def answer_accuracy_evaluator(self, run, example) -> dict:
        """
        Evaluate the accuracy of the student's answer compared to the reference answer.

        Args:
            run: The run containing the student's answer.
            example: The example containing the reference answer.

        Returns:
            dict: Evaluation result with the accuracy score.
        """
        score = self.evaluate(
            metric=hub.pull("langchain-ai/rag-answer-vs-reference"),
            inputs={
                "question": example.inputs["question"],
                "correct_answer": example.outputs["answer"],
                "student_answer": run.outputs["answer"],
            },
        )
        return {"key": "answer_v_reference_score", "score": score}

    def answer_hallucination_evaluator(self, run, example) -> dict:
        """
        Evaluate the degree of hallucination in the student's answer.

        Args:
            run: The run containing the student's answer.
            example: The example for context.

        Returns:
            dict: Evaluation result with the hallucination score.
        """
        score = self.evaluate(
            metric=hub.pull("langchain-ai/rag-answer-hallucination"),
            inputs={
                "documents": run.outputs["contexts"],
                "student_answer": run.outputs["answer"],
            },
        )
        return {"key": "answer_hallucination_score", "score": score}

    def answer_helpfulness_evaluator(self, run, example) -> dict:
        """
        Evaluate the helpfulness of the student's answer.

        Args:
            run: The run containing the student's answer.
            example: The example for context.

        Returns:
            dict: Evaluation result with the helpfulness score.
        """
        score = self.evaluate(
            metric=hub.pull("langchain-ai/rag-answer-helpfulness"),
            inputs={
                "question": example.inputs["question"],
                "student_answer": run.outputs["answer"],
            },
        )
        return {"key": "answer_helpfulness_score", "score": score}

    def document_relevancy_evaluator(self, run, example) -> dict:
        """
        Evaluate the relevancy of the retrieved documents.

        Args:
            run: The run containing the retrieved contexts.
            example: The example for context.

        Returns:
            dict: Evaluation result with the document relevancy score.
        """
        score = self.evaluate(
            metric=hub.pull("langchain-ai/rag-document-relevance"),
            inputs={
                "question": example.inputs["question"],
                "documents": run.outputs["contexts"],
            },
        )
        return {"key": "document_relevance", "score": score}
