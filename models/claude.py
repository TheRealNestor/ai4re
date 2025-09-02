from models.base_model import BaseModel

import anthropic

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key for Anthropic is not set. Please set the ANTHROPIC_API_KEY environment variable."
    )


class Claude(BaseModel):
    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-latest",
        temperature: float = 0,
        system_prompt: str = "You are a helpful assistant.",
    ):
        super().__init__(model_name, temperature, system_prompt)
        self.client = anthropic.Anthropic(api_key=API_KEY)

    # NOTE: The temperature range is [0;1] for claude.
    @BaseModel.temperature.setter
    def temperature(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self._temperature = value

    def query(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        try:
            response = self.client.messages.create(
                model=self.model_name,  # Use the configured model name
                max_tokens=4000,  # Add required max_tokens parameter
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(response, "content"):
                if isinstance(response.content, list):
                    return "".join(
                        block.text
                        for block in response.content
                        if hasattr(block, "text")
                    )
                else:
                    return str(response.content)
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
