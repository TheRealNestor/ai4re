# abstract dataclass 
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str, temperature: float = 0.7):
        self._model_name = model_name
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value
            
    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self._temperature = value

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Query the model with a prompt.

        Args:
            prompt (str): The input prompt to send to the model.
        Returns:
            str: The model's response to the prompt.
        """
        pass

