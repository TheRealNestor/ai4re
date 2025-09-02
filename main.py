import os
import json

from models.base_model import BaseModel
from models.gpt import GPT
from models.claude import Claude
from models.gemini import Gemini
from models.deepseek import DeepSeek
from models.grok import Grok
from models.mistral import MistralModel
from models.qwen import Qwen
from models.llama import Llama

import pandas as pd
from typing import Dict, List, Optional

from dataclasses import dataclass

import tqdm
import re # cleaning text

@dataclass
class RequirementResult:
    original_requirement: str
    requirement_type: str
    model_name: str
    score_response: str  # Store the full JSON response from scoring agent as string
    overall_score: Optional[int] = None  # Score from scoring agent
    refined_requirement: Optional[str] = (
        None  # Refined requirement from refinement agent
    )
    refined_response: Optional[str] = None  # Full response from refinement agent
    refined_score: Optional[int] = None  # Score from refinement agent, if applicable
    refined_score_raw_response: Optional[str] = (
        None  # Full JSON response from refinement agent, if applicable
    )


def get_prompt(prompt_name: str, prompt_dir: str = "prompts") -> str:
    prompt_path: str = os.path.join(prompt_dir, prompt_name)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"Prompt file '{prompt_name}' not found in '{prompt_dir}'."
        )

    with open(prompt_path, "r") as file:
        return file.read().strip()


def clean_text(text: str) -> str:
    """Clean corrupted characters from text."""
    if not isinstance(text, str):
        return str(text)

    # Common character encoding fixes
    replacements = {
        "â€œ": '"',  # Left double quotation mark
        "â€": '"',  # Right double quotation mark
        "â€™": "'",  # Right single quotation mark
        "â€˜": "'",  # Left single quotation mark
        'â€"': "—",  # Em dash
        'â€"': "–",  # En dash
        "Â": "",  # Non-breaking space artifacts
    }

    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    # Remove any remaining non-ASCII characters that might cause issues
    cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)

    return cleaned.strip()


def df_from_csv_fraction(
    csv_file: str, fraction: float = 1.0, random_state: int = 42
) -> pd.DataFrame:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    df = pd.read_csv(csv_file)
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=random_state)
    return df


def df_from_csv_n(csv_file: str, n: int, random_state: int = 42) -> pd.DataFrame:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    df = pd.read_csv(csv_file)
    df = df.sample(n=n, random_state=random_state)
    return df

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


# factory function for creating models
def create_model(
    model_class: BaseModel,
    sys_prompt: str,
    model_name: Optional[str] = None,
    temperature: float = 0.25,
) -> BaseModel:
    """Factory function to create models with common parameters."""
    params = {"system_prompt": sys_prompt, "temperature": temperature}
    if model_name:
        params["model_name"] = model_name
    return model_class(**params)


def extract_basic_score(response: dict) -> Optional[int]:
    """
    Extract just the overall score
    Returns score
    """
    try:
        # Try different common score field names
        score_fields = [
            "overall_score",
            "overall_quality_score",
            "score",
            "final_score",
        ]

        for field in score_fields:
            if field in response:
                score_value = response[field]
                # Handle both integer and percentage string formats
                if isinstance(score_value, str) and score_value.endswith("%"):
                    return int(score_value.rstrip("%"))
                elif isinstance(score_value, (int, float)):
                    return int(score_value)

        return None
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None


def refine_requirement(
    result: RequirementResult, refinement_model: BaseModel, refinement_input: str
) -> RequirementResult:
    """
    Refines a requirement by passing the original requirement and full scoring response
    """
    # Create refinement prompt with original requirement and full scoring context
    prompt = refinement_input.replace("{{ REQUIREMENT }}", result.original_requirement)
    prompt = prompt.replace("{{ SCORING_ANALYSIS }}", result.raw_response)

    refined_response = refinement_model.query_with_retry(prompt)

    result.refined_response = refined_response
    # Optionally extract just the refined requirement text if needed
    # result.refined_requirement = extract_refined_requirement(refined_response)

    return result


def score_requirements():
    csv_file: str = "data/software-requirements-dataset/requirements.csv"
    requirements_df = df_from_csv_n(csv_file, n=100, random_state=42)

    requirements_df["Requirement"] = requirements_df["Requirement"].apply(clean_text)

    print(f"Loaded {len(requirements_df)} requirements from {csv_file}")
    print(requirements_df.head())

    scoring_prompt: str = get_prompt(
        "mixture_of_opinions_v2.txt", prompt_dir="prompts/scoring"
    )

    scoring_models: List[BaseModel] = [
        # MistralModel(
        #     model_name="magistral-medium-2507",  #  frontier-class reasoning model released July 2025.
        #     system_prompt=scoring_prompt,
        # ),
        MistralModel(
            model_name="mistral-medium-latest", # could go with "mistral-medium-2508" also
            system_prompt=scoring_prompt
        ),



        # GPT(
        #     model_name="gpt-5",  # Sometimes a thinking/reasoning model for complex prompts or when manually activated? Released in august 2025
        #     system_prompt=scoring_prompt,
        #     temperature=1, # Only default value of 1 is supported.
        # ),
        # GPT(
        #     model_name="gpt-4.1",
        #     system_prompt=scoring_prompt,
        # ),
        GPT(
            model_name="gpt-5-mini",
            system_prompt=scoring_prompt,
            temperature=1
        ),
        GPT(
            model_name="gpt-5-nano",
            system_prompt=scoring_prompt,
            temperature=1
        )




        # Claude(
        #     model_name="claude-sonnet-4-20250514",
        #     system_prompt=scoring_prompt,
        # ),
        
        
        # Gemini(
        #     model_name="gemini-2.5-pro",
        #     system_prompt=scoring_prompt
        # ),
        # Gemini(
        #     model_name="gemini-2.5-flash",
        #     system_prompt=scoring_prompt
        # ),
        # Gemini(
        #     model_name="gemini-2.5-flash-lite",
        #     system_prompt=scoring_prompt
        # )


    ]

    # Create output directory
    os.makedirs("results", exist_ok=True)

    for model in scoring_models:
        print(f"Processing with model: {model.model_name}")
        model_results = []

        for idx, row in tqdm.tqdm(requirements_df.iterrows(), desc=f"Scoring with {model.model_name}", total=len(requirements_df)):
            req = row["Requirement"]
            req_type = row["Type"]

            response = model.query_with_retry(req)
            if not response:
                print(
                    f"No response for requirement: {req} from model: {model.model_name}"
                )
                continue

            cleaned_response = clean_json_output(response)

            if not cleaned_response:
                print(
                    f"Failed to parse JSON response for requirement: {req} from model: {model.model_name}"
                )
                continue
            
            overall_score = extract_basic_score(cleaned_response)

            result = RequirementResult(
                original_requirement=req,
                requirement_type=req_type,
                model_name=model.model_name,
                score_response=cleaned_response,
                overall_score=overall_score,
                refined_requirement=None,
                refined_response=None,
                refined_score=None,
                refined_score_raw_response=None,
            )

            model_results.append(result)

        # Save this model's results immediately
        safe_model_name = model.model_name.replace('-', '_').replace('.', '_')
        output_file_path = f"results/{safe_model_name}_scores.json"
        with open(output_file_path, "w") as f:
            json.dump([result.__dict__ for result in model_results], f, indent=4)
        print(f"Results for {model.model_name} written to {output_file_path}")


if __name__ == "__main__":

    score_requirements()

    # # REFINE the requirements
    # refinement_prompt: str = get_prompt("refinement_v2.txt", prompt_dir="prompts/refinement")

    # # TODO: Use this instead
    # refinement_prompts: List[str] = [
    #     "refinement_v2.txt", "few_shot_refinement_v0.txt", "generated_knowledge_v0.txt", "question_answer_v0.txt", "reflective_critic_v0.txt", "rules_v1.txt",
    #     "tree_of_thought_v0.txt", "chain_of_thought_v0.txt", "chain_of_density_v0.txt", "directional_stimulus_v0.txt"
    # ]

    # refinement_models: List[BaseModel] = [
    #     MistralModel(system_prompt=refinement_prompt, temperature=0.25),
    # ]

    # refined_results: List[RequirementResult] = []

    # for model in refinement_models:
    #     for req, req_results in results.items():
    #         for result in req_results:  # Each model should have a result for each requirement
    #             print(f"Refining requirement with {model.model_name}: {req[:50]}...")

    #             refinement_input = (
    #                 f"Original requirement: {result.original_requirement}\n"
    #                 f"Scores: {result.original_score}\n"
    #                 "Please refine the requirement based on the scores and justifications provided."
    #             )

    #             refined_result = refine_requirement(
    #                 result, model, refinement_input
    #             )

    # # Write the final results including refinements to a JSON file
    # refined_output_path = "results/refined_requirements.json"
    # print(len(refined_results))
    # with open(refined_output_path, "w") as f:
    #     json.dump([result.__dict__ for result in refined_results], f, indent=4)
    # print(f"Refined results written to {refined_output_path}")

    # # Create summary report
    # summary_output_path = "results/refinement_summary.json"
    # summary_data = []

    # for result in refined_results:
    #     if (
    #         result.refined_requirement
    #         and result.original_score
    #         and result.refined_score
    #     ):
    #         original_overall = result.original_score.get("overall_quality_score", 0)
    #         refined_overall = result.refined_score.get("overall_quality_score", 0)

    #         summary_data.append(
    #             {
    #                 "original_requirement": result.original_requirement,
    #                 "refined_requirement": result.refined_requirement,
    #                 "original_score": original_overall,
    #                 "refined_score": refined_overall,
    #                 "model_name": result.model_name,
    #                 "requirement_type": result.requirement_type,
    #             }
    #         )

    # with open(summary_output_path, "w") as f:
    #     json.dump(summary_data, f, indent=4)
    # print(f"Summary report written to {summary_output_path}")

    # Define different input prompts

    # Now we can refine the requirements based on the scores and justifications.
    # OUTPUT FORMAT:
    # {original requirement, type, original score, justifications, refined requirement, refined score}
    # Maybe do EARS syntax for refined requirement

    # scoring_prompt = get_prompt("scoring_v4.txt", prompt_dir="prompts/scoring")
    # scoring_prompt = get_prompt("mixture_of_opinions_v2.txt", prompt_dir="prompts/scoring")
    # model = GPT(
    #     model_name="gpt-5",
    #     temperature=0)


    # example_requirement_good: str = "The system shall be in compliance with IP level IP44 as defined in IEC 60529."
    # example_requirement_bad: str = "The system shall comply with EN 61800-5-1:2007"

    # response = model.query(example_requirement_good)
    # print(f"Response: {response}")
    # print("-------")
    # response = model.query(example_requirement_bad)
    # print(f"Response: {response}")
    # print("-------")
