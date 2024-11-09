"""
Each node in our graph is simply a function that:
(1) Take state as an input
(2) Modifies state
(3) Write the modified state to the state schema (dict)
"""
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from backend.models.chat_models import RetrievalAugmentedGenerator, RetrievalGrader
from backend.models.retriever import Retriever


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    retriever = Retriever()
    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    rag = RetrievalAugmentedGenerator()
    return {"generation": rag.invoke(documents=documents, question=question), "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    retrieval_grader = RetrievalGrader()

    # Score each doc
    filtered_documents = []
    web_search = "No"
    for document in documents:
        grade = retrieval_grader.invoke(document=document, question=question)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_documents.append(document)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_documents
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_documents, "web_search": web_search}


def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    web_search_tool = TavilySearchResults(k=3)
    web_searched_documents = web_search_tool.invoke({"query": question})
    web_results = "\n".join([doc["content"] for doc in web_searched_documents])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}
