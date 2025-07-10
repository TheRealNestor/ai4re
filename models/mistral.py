from models.base_model import BaseModel

from mistralai import Mistral

import os 
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key for Mistral is not set. Please set the MISTRAL_API_KEY environment variable."
    )

class MistralModel(BaseModel):

    def __init__(
        self,
        model_name: str = "mistral-medium-latest", # Mistral medium 3, possibly worth trying "Mistral large" too and "Mistral small 3.2" (which is also open and cheaper)
        temperature: float = 0.7,
        system_prompt: str = BaseModel.__init__.__defaults__[
            1
        ],  # Default system prompt from BaseModel
    ):
        super().__init__(model_name, temperature, system_prompt)
        self.client = Mistral(api_key=API_KEY)

    def query(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
