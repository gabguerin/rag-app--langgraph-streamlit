import json

from langsmith import Client

from backend.evaluation.generate_test_set import TEST_SET_FILENAME

LANGSMITH_DATASET_NAME = "OXFAM_dataset"

if __name__ == "__main__":
    test_set = {
        "inputs": [],
        "outputs": [],
    }
    with open(
        f"backend/evaluation/datasets/{TEST_SET_FILENAME}.jsonl", encoding="utf-8"
    ) as file:
        for line in file.readlines():
            example = json.loads(line)
            test_set["inputs"].append({"question": example["question"]})
            test_set["outputs"].append({"answer": example["reference_answer"]})

    client = Client()

    # Define dataset: these are your test cases
    dataset = client.create_dataset(LANGSMITH_DATASET_NAME)
    client.create_examples(
        inputs=test_set["inputs"],
        outputs=test_set["outputs"],
        dataset_id=dataset.id,
    )
