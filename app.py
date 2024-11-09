from langchain.rag_graph.graph import graph
import os
import getpass


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


os.environ["TAVILY_API_KEY"] = "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"
# _set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
# _set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

if __name__ == "__main__":
    inputs = {
        "question": "Give a summary of the key results of TotalEnergies in 2023",
        "max_retries": 3
    }
    for event in graph.stream(inputs, stream_mode="values"):
        print(event)
