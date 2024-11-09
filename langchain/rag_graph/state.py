"""
The graph state schema contains keys that we want to:
    - Pass to each node in our graph
    - Optionally, modify in each node of our graph
"""
import operator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from typing import List, Annotated


class State(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: BaseMessage  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[Document]  # List of retrieved documents
