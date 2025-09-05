from models.base_model import BaseModel

from google import genai
from google.genai import types


import json
import re

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key for Gemini is not set. Please set the GEMINI_API_KEY environment variable."
    )


class Google(BaseModel):

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0,
        system_prompt: str = BaseModel.__init__.__defaults__[
            1
        ],  # Default system prompt from BaseModel
    ):
        super().__init__(model_name, temperature, system_prompt)
        self.client = genai.Client(api_key=API_KEY)

    def query(self, prompt: str) -> str:
        """
        Query the OpenAI GPT model with a prompt.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            str: The model's response to the prompt.

        Raises:
            ValueError: If the prompt is empty.
            Exception: If an error occurs during the API call.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=self.temperature,
                ),
                contents=prompt,
            )
            content = response.candidates[0].content if response.candidates else None
            # If content is an object, extract its text attribute
            if content:
                # Extract text from Content object
                if hasattr(content, "parts") and content.parts:
                    # Assume first part contains the main text
                    part = content.parts[0]
                    content_text = getattr(part, "text", None)
                elif hasattr(content, "text"):
                    content_text = content.text
                elif isinstance(content, str):
                    content_text = content
                else:
                    print("Unknown content type:", type(content))
                    return None
                # Extract JSON from Markdown or text
                match = re.search(r"```json\s*(\{.*\})\s*```", content_text, re.DOTALL)
                if not match:
                    match = re.search(r"(\{.*\})", content_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    return json_str
                else:
                    print("No JSON found in response.")
                    return None
            return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
