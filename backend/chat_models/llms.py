import json
from typing import List

import ollama
import yaml

MODEL_NAME = "llama3.2:3b-instruct-fp16"


class LLM:
    def __init__(
        self,
        prompt: dict[str],
        prompt_inputs: List[str],
        instructions: str | None,
        format_json: bool,
        model_name: str = MODEL_NAME,
        language: str = "fra",
    ):
        self.prompt = prompt[language]
        self.prompt_inputs = prompt_inputs
        self.instructions = instructions or ""
        self.format_json = format_json

        self.model_name = model_name

    def invoke(self, inputs: dict[str]):
        if len(set(self.prompt_inputs) - set(list(inputs.keys()))) > 0:
            raise ValueError(f"Input dict should contain {self.prompt_inputs} keys")
        formatted_prompt = self.prompt.format(**inputs)

        print(f"Formatted prompt: {formatted_prompt}")
        result = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.instructions,
                },
                {
                    "role": "user",
                    "content": formatted_prompt,
                },
            ],
        )["message"]["content"]
        print(f"Result {result}")

        if self.format_json:
            return json.loads(result)
        return result


# Load model configs
with open("backend/chat_models/model_configs.yml", "r", encoding="utf-8") as f:
    model_configs = yaml.safe_load(f)

# Instantiate chat models from config
rag_model = LLM(**model_configs["retrieval_augmented_generator"])
summarizer = LLM(**model_configs["multi_modal_summarizer"])

rewriter = LLM(**model_configs["question_rewriter"])

retrieval_grader = LLM(**model_configs["retrieval_grader"])
hallucination_grader = LLM(**model_configs["hallucination_grader"])
answer_grader = LLM(**model_configs["answer_grader"])
router = LLM(**model_configs["router"])
