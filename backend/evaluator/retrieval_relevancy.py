from langchain import hub
from langchain_openai import ChatOpenAI

grade_prompt_retrieval_relevancy = hub.pull("langchain-ai/rag-document-relevance")

def retrieval_relevancy_evaluator(run, example) -> dict:
    """
    A simple evaluator for document relevance
    """

    # RAG inputs
    input_question = example.inputs["input_question"]
    contexts = run.outputs["contexts"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_retrieval_relevancy | llm

    # Get score
    score = answer_grader.invoke({"question":input_question,
                                  "documents":contexts})
    score = score["Score"]

    return {"key": "document_relevance", "score": score}
