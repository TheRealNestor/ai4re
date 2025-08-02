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
from typing import Dict, List, Optional, TypedDict, Literal, get_args, Any

from dataclasses import dataclass

QualityCriteria = Literal[
    "Unambiguous",
    "Verifiable",
    "Feasible",
    "Complete",
    "Correct",
    "Consistent",
    "Modifiable",
]


class ScoreItem(TypedDict):
    score: int
    justification: str

class ScoreCollection(TypedDict):
    scores: Dict[QualityCriteria, ScoreItem]
    overall_quality_score: int


@dataclass
class RequirementResult:
    original_requirement: str
    requirement_type: str
    model_name: str 
    original_score: Optional[ScoreCollection] = None
    refined_requirement: Optional[str] = None
    refined_score: Optional[ScoreCollection] = None


def get_prompt(prompt_name: str, prompt_dir: str = "prompts") -> str:
    prompt_path: str = os.path.join(prompt_dir, prompt_name)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file '{prompt_name}' not found in '{prompt_dir}'.")

    with open(prompt_path, 'r') as file:
        return file.read().strip()


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


def clean_and_parse_json(response: str) -> dict:
    """
    Universal function to clean markdown-wrapped JSON and parse it.
    Handles both ```json and ``` code blocks.
    """
    try:
        cleaned_response = response.strip()

        # Remove markdown code block markers if present
        if cleaned_response.startswith("```json"):
            start_idx = cleaned_response.find("```json") + 7
            end_idx = cleaned_response.rfind("```")
            if end_idx > start_idx:
                cleaned_response = cleaned_response[start_idx:end_idx].strip()
        elif cleaned_response.startswith("```"):
            start_idx = cleaned_response.find("```") + 3
            end_idx = cleaned_response.rfind("```")
            if end_idx > start_idx:
                cleaned_response = cleaned_response[start_idx:end_idx].strip()

        return json.loads(cleaned_response)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response[:200]}...")
        raise
    except Exception as e:
        print(f"Unexpected error parsing response: {e}")
        raise


def parse_scores_and_justifications(response: str) -> ScoreCollection:
    """
    Parses the model response JSON and extracts scores and justifications.
    """
    try:
        data = clean_and_parse_json(response)
        return data
    except Exception as e:
        print(f"Error parsing scores: {e}")
        # Return empty structure on error
        return {
            "scores": {
                k: {"score": 0, "justification": ""} for k in get_args(QualityCriteria)
            },
            "overall_quality_score": 0,
        }


def refine_requirement(
    result: RequirementResult, refinement_model: BaseModel, refinement_input: str
) -> RequirementResult:
    """
    Refines a requirement based on scoring feedback and returns updated RequirementResult
    """
    if result.original_score is None:
        print(
            f"No original score available for requirement: {result.original_requirement[:50]}..."
        )
        return result

    try:
        refinement_response = refinement_model.query(refinement_input)
        if not refinement_response:
            print(f"No refinement response from {refinement_model.model_name}")
            return result

        refinement_data = clean_and_parse_json(refinement_response)
        result.refined_requirement = refinement_data.get("refined_requirements")
        result.refinement_justification = refinement_data.get("justification")

        return result

    except Exception as e:
        print(f"Failed to refine requirement with {refinement_model.model_name}: {e}")
        return result


if __name__ == "__main__":

    csv_file: str = "data/software-requirements-dataset/requirements.csv"
    requirements_df = df_from_csv_n(csv_file, n=3, random_state=42)

    print(f"Loaded {len(requirements_df)} requirements from {csv_file}")
    print(requirements_df.head())

    scoring_prompt: str = get_prompt("scoring_v2.txt", prompt_dir="prompts/scoring")
    scoring_models: List[BaseModel] = [
        # MistralModel(system_prompt=scoring_prompt, temperature=0.25), # TODO: Update from free tier
        GPT(model_name="gpt-4.1", system_prompt=scoring_prompt, temperature=0.25),
        # Claude(system_prompt=scoring_prompt, temperature=0.25),
        # Gemini(system_prompt=scoring_prompt, temperature=0.25),
        # DeepSeek(system_prompt=scoring_prompt, temperature=0.25),
        # Grok(system_prompt=scoring_prompt, temperature=0.25),
        # Gwen(system_prompt=scoring_prompt, temperature=0.25),
        # Llama(system_prompt=scoring_prompt, temperature=0.25),
    ]

    results: Dict[str, List[RequirementResult]] = {} # key is the original requirement (ensures that no duplicates are processed), list because multiple models

    for model in scoring_models:
        for idx, row in requirements_df.iterrows():
            req = row["Requirement"]
            req_type = row["Type"]

            response = model.query(req)
            if not response:
                print(
                    f"No response for requirement: {req} from model: {model.model_name}"
                )
                continue

            parsed_scores = parse_scores_and_justifications(response)
            if not parsed_scores:
                print(
                    f"Failed to parse scores for requirement: {req} from model: {model.model_name}"
                )
                continue

            result = RequirementResult(
                original_requirement=req,
                requirement_type=req_type,
                model_name=model.model_name,
                original_score=parsed_scores,
            )

            if req not in results:
                results[req] = []
            results[req].append(result)

    flattened_results = []
    for req_results in results.values():
        flattened_results.extend(req_results)

    # Write the intermediate results to a JSON file
    output_file_path = "results/requirement_scores.json"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump([result.__dict__ for result in flattened_results], f, indent=4)
    print(f"Results written to {output_file_path}")

    # REFINE the requirements
    refinement_prompt: str = get_prompt("refinement_v2.txt", prompt_dir="prompts/refinement")
    refinement_models: List[BaseModel] = [
        MistralModel(system_prompt=refinement_prompt, temperature=0.25),
    ]

    refined_results: List[RequirementResult] = []

    for model in refinement_models:
        for req, req_results in results.items():
            for result in req_results:  # Each model should have a result for each requirement
                print(f"Refining requirement with {model.model_name}: {req[:50]}...")

                refinement_input = (
                    f"Original requirement: {result.original_requirement}\n"
                    f"Scores: {result.original_score}\n"
                    "Please refine the requirement based on the scores and justifications provided."
                )

                refined_result = refine_requirement(
                    result, model, refinement_input
                )

    # Write the final results including refinements to a JSON file
    refined_output_path = "results/refined_requirements.json"
    print(len(refined_results))
    with open(refined_output_path, "w") as f:
        json.dump([result.__dict__ for result in refined_results], f, indent=4)
    print(f"Refined results written to {refined_output_path}")

    # Create summary report
    summary_output_path = "results/refinement_summary.json"
    summary_data = []

    for result in refined_results:
        if (
            result.refined_requirement
            and result.original_score
            and result.refined_score
        ):
            original_overall = result.original_score.get("overall_quality_score", 0)
            refined_overall = result.refined_score.get("overall_quality_score", 0)

            summary_data.append(
                {
                    "original_requirement": result.original_requirement,
                    "refined_requirement": result.refined_requirement,
                    "original_score": original_overall,
                    "refined_score": refined_overall,
                    "model_name": result.model_name,
                    "requirement_type": result.requirement_type,
                }
            )

    with open(summary_output_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Summary report written to {summary_output_path}")




    # Define different input prompts

    # Now we can refine the requirements based on the scores and justifications.
    # OUTPUT FORMAT:
    # {original requirement, type, original score, justifications, refined requirement, refined score}
    # Maybe do EARS syntax for refined requirement

    # example_requirement_good: str = "The system shall be in compliance with IP level IP44 as defined in IEC 60529."
    # example_requirement_bad: str = "The system shall comply with EN 61800-5-1:2007"

    # response = model.query(example_requirement_good)
    # response = model.query(example_requirement_bad)

    # Let's extract the score and justifications, use these to refine the requirement.

    # print(f"Response: {response}")
