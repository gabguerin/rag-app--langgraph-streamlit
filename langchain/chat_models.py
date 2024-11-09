import json
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b-instruct-fp16"


def format_documents(documents: List[Document]):
    return "\n\n".join(doc.page_content for doc in documents)


class RetrievalAugmentedGenerator:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)
        self.rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

    def invoke(self, documents: List[Document], question: str):
        documents_txt = format_documents(documents)
        rag_prompt_formatted = self.rag_prompt.format(context=documents_txt, question=question)
        return self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])


class RetrievalGrader:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
        self.prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

    def invoke(self, document: Document, question: str):
        prompt_formatted = self.prompt.format(
            document=document, question=question
        )
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=prompt_formatted)]
        )
        print(json.loads(result.content))
        return json.loads(result.content)

class HallucinationGrader:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """
You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""
        self.prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

    def invoke(self, documents: List[Document], generation: str):
        prompt_formatted = self.prompt.format(
            documents=format_documents(documents), generation=generation
        )
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=prompt_formatted)]
        )
        return json.loads(result.content)

class AnswerGrader:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""
        self.prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

    def invoke(self, question: str, generation: str):
        prompt_formatted = self.prompt.format(
            question=question, generation=generation
        )
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=prompt_formatted)]
        )
        return json.loads(result.content)

class Router:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to TotalEnergies results in 2023.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

    def invoke(self, question: str):
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=question)]
        )
        return json.loads(result.content)
