from typing import List, Any

import openai
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langsmith import traceable

from backend.chat_models.llms import LLM, rag_model
from backend.utils import format_documents
from backend.vectorstore import MultiModalVectorstore


class RagEvaluator:

    def __init__(self, student_llm: LLM, db: MultiModalVectorstore):
        self._student_llm = student_llm
        self._db = db

        self._evaluator_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    @traceable()
    def retrieve_docs(self, question: str) -> List[Document]:
        return self._db.retrieve_documents(question)

    @traceable()
    def invoke_llm(self, question: str, documents: List[Document]):
        response = self._student_llm.invoke(
            inputs={"context": format_documents(documents), "question": question}
        )

        # Evaluators will expect "answer" and "contexts"
        return {
            "answer": response,
            "contexts": [doc.page_content for doc in documents],
        }

    @traceable()
    def get_answer(self, question: str):
        documents = self.retrieve_docs(question)
        return self.invoke_llm(question, documents)

    def predict_rag_answer(self, example: dict):
        """Use this for answer evaluation"""
        response = self.get_answer(example["input_question"])
        return {"answer": response["answer"]}

    def predict_rag_answer_with_context(self, example: dict):
        """Use this for evaluation of retrieved documents and hallucinations"""
        response = self.get_answer(example["input_question"])
        return {"answer": response["answer"], "contexts": response["contexts"]}

    def evaluate(self, metric: Any, inputs: dict[str, str]):
        # Structured prompt
        answer_grader = metric | self._evaluator_llm

        # Run evaluator
        score = answer_grader.invoke(**inputs)
        return score["Score"]
