import json

def clean_json_output(response: str):
    """Clean and parse JSON output from model response."""
    if not response:
        return None

    try:
        # Try to extract JSON from response if it's wrapped in markdown or other text
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx : end_idx + 1]
            return json.loads(json_str)
        else:
            # If no JSON brackets found, try parsing the whole response
            return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response length: {len(response)} characters")
        print("=" * 80)
        print("FULL RESPONSE BEING PARSED:")
        print("=" * 80)
        print(response)
        print("=" * 80)

        # Also show just the JSON part if we found brackets
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx : end_idx + 1]
            print("EXTRACTED JSON STRING:")
            print("=" * 80)
            print(json_str)
            print("=" * 80)

        return None
    except Exception as e:
        print(f"Unexpected error parsing JSON: {e}")
        print("=" * 80)
        print("FULL RESPONSE BEING PARSED:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        return None


def is_valid_json(s: str) -> bool:
    """
    Check if a string is valid JSON.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is valid JSON, False otherwise.
    """

    if clean_json_output(s) is not None:
        return True
    return False
