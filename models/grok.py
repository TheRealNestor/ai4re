from models.base_model import BaseModel

from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GROK_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key for Grok is not set. Please set the GROK_API_KEY environment variable."
    )

class Grok(BaseModel):
    def __init__(
        self,
        model_name: str = "grok-3-mini",
        temperature: float = 0,
        system_prompt: str = BaseModel.__init__.__defaults__[1],  # Default system prompt from BaseModel
    ):
        super().__init__(model_name, temperature, system_prompt)
        self.client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1")

    def query(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
