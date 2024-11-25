from langchain import hub
from langchain_openai import ChatOpenAI

grade_prompt_answer_helpfulness = prompt = hub.pull(
    "langchain-ai/rag-answer-helpfulness"
)


def answer_helpfulness_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer helpfulness
    """

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["input_question"]
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_answer_helpfulness | llm

    # Run evaluator
    score = answer_grader.invoke(
        {"question": input_question, "student_answer": prediction}
    )
    score = score["Score"]

    return {"key": "answer_helpfulness_score", "score": score}
