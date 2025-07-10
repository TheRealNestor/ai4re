import os 

from models.gpt import GPT
from models.claude import Claude
from models.gemini import Gemini
from models.deepseek import DeepSeek
from models.grok import Grok
from models.mistral import MistralModel

def get_prompt(prompt_name: str, prompt_dir: str = "prompts") -> str:
    prompt_path: str = os.path.join(prompt_dir, prompt_name)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file '{prompt_name}' not found in '{prompt_dir}'.")
    
    with open(prompt_path, 'r') as file:
        return file.read().strip()




if __name__ == "__main__":

    sys_prompt_scoring: str = get_prompt("scoring_v2.txt")
    # model = GPT(system_prompt=sys_prompt_scoring)
    # model = Claude(system_prompt=sys_prompt_scoring)
    # model = Gemini(system_prompt=sys_prompt_scoring, temperature=0.3)
    model = MistralModel(system_prompt=sys_prompt_scoring, temperature=0.42) # Free plan for now, may hit rate limits

    # Need to set up credits for these currently, or run locally when possible
    # model = DeepSeek(system_prompt=sys_prompt_scoring)
    # model = Grok(system_prompt=sys_prompt_scoring)

    example_requirement_good: str = "The system shall be in compliance with IP level IP44 as defined in IEC 60529."
    example_requirement_bad: str = "The system shall comply with EN 61800-5-1:2007"

    response = model.query(example_requirement_good)
    # response = model.query(example_requirement_bad)

    # Let's extract the score and justifications, use these to refine the requirement. 

    # if score < 0.85
    # refine it
    # else do nothing
    print(f"Response: {response}")




