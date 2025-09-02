from models.base_model import BaseModel

from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key for Gemini is not set. Please set the GEMINI_API_KEY environment variable."
    )


class Gemini(BaseModel):

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0,
        system_prompt: str = BaseModel.__init__.__defaults__[
            1
        ],  # Default system prompt from BaseModel
    ):
        super().__init__(model_name, temperature, system_prompt)
        self.client = OpenAI(
            api_key=API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def query(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=65535,  # This should be default, but I had outputs that were truncated.
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
