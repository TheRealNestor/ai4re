from models.base_model import Model

from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


# override the Model superclass (ABC) to implement the ChatGPT model
class ChatGPTModel(Model):
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.client = OpenAI(api_key=API_KEY)

    def query(self, prompt: str) -> str:
        """
        Query the ChatGPT model with a prompt.

        Args:
            prompt (str): The input prompt to send to the model.
            
        Returns:
            str: The model's response to the prompt.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        try:    
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
