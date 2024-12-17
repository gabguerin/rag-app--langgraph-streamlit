import json
from typing import List

import yaml
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


# Load env variables
load_dotenv()


class LLM:
    def __init__(
        self,
        prompt: dict[str],
        prompt_inputs: List[str],
        instructions: str | None,
        format_json: bool,
        language: str = "fra",
    ):
        self.prompt = prompt[language]
        self.prompt_inputs = prompt_inputs
        self.instructions = instructions or ""
        self.format_json = format_json

        self.chat_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )
        if self.format_json:
            self.chat_model = self.chat_model.bind(
                response_format={"type": "json_object"}
            )

    def invoke(self, inputs: dict[str, str]) -> str:
        if len(set(self.prompt_inputs) - set(list(inputs.keys()))) > 0:
            raise ValueError(f"Input dict should contain {self.prompt_inputs} keys")
        formatted_prompt = self.prompt.format(**inputs)

        result = self.chat_model.invoke(
            input=[
                (
                    "system",
                    self.instructions,
                ),
                (
                    "user",
                    formatted_prompt,
                ),
            ],
        ).content

        if self.format_json:
            return json.loads(result)
        return result


# Load model configs
with open("backend/chat_models/system_instructions.yml", "r", encoding="utf-8") as f:
    model_configs = yaml.safe_load(f)

# Instantiate chat models from config
rag_model = LLM(**model_configs["retrieval_augmented_generator"])
summarizer = LLM(**model_configs["multi_modal_summarizer"])

rewriter = LLM(**model_configs["question_rewriter"])

retrieval_grader = LLM(**model_configs["retrieval_grader"])
hallucination_grader = LLM(**model_configs["hallucination_grader"])
answer_grader = LLM(**model_configs["answer_grader"])
router = LLM(**model_configs["router"])
