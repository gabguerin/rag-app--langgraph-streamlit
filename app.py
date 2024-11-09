from backend.rag_graph.workflow import graph
import os
import getpass


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

if __name__ == "__main__":
    inputs = {"question": "What are the types of agent memory?", "max_retries": 3}
    for event in graph.stream(inputs, stream_mode="values"):
        print(event)