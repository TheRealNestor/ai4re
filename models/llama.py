from models.base_model import BaseModel

from together import Together

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key for Together.ai is not set. Please set the TOGETHER_API_KEY environment variable."
    )


class Llama(BaseModel):

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # meta-llama/Llama-4-Scout-17B-16E-Instruct
        temperature: float = 0,
        system_prompt: str = BaseModel.__init__.__defaults__[
            1
        ],  # Default system prompt from BaseModel
    ):
        super().__init__(model_name, temperature, system_prompt)
        self.client = Together(api_key=API_KEY)

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
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
