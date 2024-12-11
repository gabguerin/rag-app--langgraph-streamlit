import pandas as pd
from dotenv import load_dotenv

from giskard import llm
from giskard.rag import generate_testset, KnowledgeBase
from giskard.rag.question_generators import (
    simple_questions,
    complex_questions,
    double_questions,
    situational_questions,
)
from langsmith import Client

from backend.vectorstore import MultiModalVectorstore

load_dotenv()

# Optional, setup a model (default LLM is gpt-4o, default embedding model is text-embedding-3-small)
llm.set_llm_model("gpt-4o")
llm.set_embedding_model("text-embedding-3-small")

TEST_SET_FILENAME = "OXFAM_dataset"

if __name__ == "__main__":
    db = MultiModalVectorstore()

    # Load your datasets and initialize the KnowledgeBase
    knowledge_base_df = pd.DataFrame(
        db.get_all_documents_in_vectorstore(), columns=["document"]
    )

    knowledge_base = KnowledgeBase.from_pandas(knowledge_base_df, columns=["document"])

    # Generate a test set with 10 questions & answers for each question types (this will take a while)
    test_set = generate_testset(
        knowledge_base,
        num_questions=100,
        language="fr",
        agent_description="Expert dans les rapports financiers et autres informations relatives aux entreprises du SBF120 en France.",  # helps generating better questions
        question_generators=[
            simple_questions,
            complex_questions,
            situational_questions,
            double_questions,
        ],
    )

    # Save the generated testset
    test_set.save(f"backend/evaluation/datasets/{TEST_SET_FILENAME}.jsonl")

    test_set_df = test_set.to_pandas()
    client = Client()
