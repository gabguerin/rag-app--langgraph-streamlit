from langgraph.constants import END
from langgraph.graph import StateGraph

from backend.rag_graph.edges import (
    route_question,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)
from backend.rag_graph.state import State
from backend.rag_graph.nodes import (
    web_search,
    retrieve,
    grade_documents,
    generate,
    rewrite,
    generate_question_not_relevant,
    # generate_question_not_relevant,
)

workflow = StateGraph(State)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# SIMPLE GRAPH

simple_workflow = StateGraph(State)

# simple_workflow.set_entry_point("retrieve")

simple_workflow.add_node("retrieve", retrieve)
simple_workflow.add_node("rewrite", rewrite)
simple_workflow.add_node("generate", generate)
simple_workflow.add_node(
    "generate_question_not_relevant", generate_question_not_relevant
)

# Build graph
simple_workflow.set_conditional_entry_point(
    route_question,
    {
        "yes": "retrieve",
        "no": "question_not_relevant",
    },
)

simple_workflow.add_edge("retrieve", "rewrite")
simple_workflow.add_edge("question_not_relevant", "generate_question_not_relevant")
simple_workflow.add_edge("rewrite", "generate")
simple_workflow.add_edge("generate", END)
simple_workflow.add_edge("generate_question_not_relevant", END)

# Compile
graph = simple_workflow.compile()
