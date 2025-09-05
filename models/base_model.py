from abc import ABC, abstractmethod
import time
import random

from util.json_utils import is_valid_json

class BaseModel(ABC):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0,
        system_prompt: str = "You are a helpful assistant.",
        max_retries: int = 10,
        retry_delay: float = 0.2,
    ):
        self._model_name = model_name
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._max_retries = max_retries
        self._retry_delay = retry_delay

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

    def query_with_retry(self, prompt: str) -> str:
        """
        Query the model with retry logic for handling temporary failures.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            str: The model's response to the prompt, or None if all retries failed.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        for attempt in range(self._max_retries):
            try:
                response = self.query(prompt)
                if response and is_valid_json(response):
                    return response
                else:
                    # Treat None or invalid JSON as retryable error
                    print(
                        f"Attempt {attempt + 1} failed for {self.model_name}: Invalid or empty response."
                    )
            except Exception as e:
                error_msg = str(e).upper()
                non_retryable_errors = [
                    "MAX_TOKENS",
                ]
                retryable_errors = [
                    "CONNECTION ERROR",
                    "INTERNAL SERVER ERROR",
                    "500",
                    "502",
                    "503",
                    "504",
                    "UNAVAILABLE",
                    "TIMEOUT",
                    "RATE LIMIT",
                    "COULD NOT EXTRACT TEXT FROM RESPONSE",
                    "PARTS=NONE",
                    "TRUNCATED_JSON",
                ]

                print(
                    f"Attempt {attempt + 1} failed for {self.model_name}: {e}. Retrying in {self._retry_delay:.1f}s..."
                )

                is_non_retryable = any(err in error_msg for err in non_retryable_errors)
                is_retryable = any(err in error_msg for err in retryable_errors)

                if is_non_retryable:
                    print(f"Non-retryable error for {self.model_name}: {e}")
                    return None
                elif not is_retryable or attempt == self._max_retries - 1:
                    print(f"Final attempt failed for {self.model_name}: {e}")
                    return None

            # Exponential backoff + jitter
            delay = self._retry_delay * (2**attempt) + random.uniform(0, 1)
            time.sleep(delay)
            continue

        return None

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
