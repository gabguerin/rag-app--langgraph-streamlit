import pandas as pd

from giskard import llm
from giskard.rag import generate_testset, KnowledgeBase
from langsmith import Client

from backend.vectorstore import MultiModalVectorstore

# Optional, setup a model (default LLM is gpt-4o, default embedding model is text-embedding-3-small)
llm.set_llm_model("gpt-4o")
llm.set_embedding_model("text-embedding-3-small")

DATASET_NAME = "OXFAM Dataset"

if __name__=="__main__":
    db = MultiModalVectorstore()

    # Load your data and initialize the KnowledgeBase
    knowledge_base_df = pd.DataFrame(db.get_all_documents_in_vectorstore(), columns=["document"])

    knowledge_base = KnowledgeBase.from_pandas(knowledge_base_df, columns=["document"])

    # Generate a test set with 10 questions & answers for each question types (this will take a while)
    test_set = generate_testset(
        knowledge_base,
        num_questions=60,
        language='fr',
        agent_description="Expert dans les rapports financiers et autres informations relatives aux entreprises du SBF120 en France.", # helps generating better questions
    )

    # Save the generated testset
    test_set.save(f"./data/{DATASET_NAME}.jsonl")
    test_set_df = test_set.to_pandas()

    client = Client()

    # Define dataset: these are your test cases
    dataset = client.create_dataset(DATASET_NAME)
    client.create_examples(
        inputs=[
            {"question": question for question in test_set_df["question"].values},
        ],
        outputs=[
            {"answer": answer for answer in test_set_df["reference_answer"].values},
        ],
        dataset_id=dataset.id,
        metadata=test_set_df["metadata"].values
    )