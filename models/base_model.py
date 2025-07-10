from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self._model_name = model_name
        self._temperature = temperature
        self._system_prompt = system_prompt

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
        if not (0.0 <= value <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self._temperature = value

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        if not value:
            raise ValueError("System prompt cannot be empty.")
        self._system_prompt = value

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Query the model with a prompt.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            str: The model's response to the prompt.

        Raises:
            ValueError: If the prompt is empty.
            Exception: If an error occurs during the API call.
        """
        pass
