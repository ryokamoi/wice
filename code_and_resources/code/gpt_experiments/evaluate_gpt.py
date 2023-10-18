from pathlib import Path
from typing import Literal
from tqdm import tqdm

from tap import Tap
from llm_wrapper import llm_api
from easy_io import read_jsonl, dump_jsonl, dump_json

from gpt_experiments.prompts import get_gpt_prompt


class GPTEvalTap(Tap):
    model: Literal["gpt-3.5-turbo-0613", "gpt-4-0613"]
    claim_subclaim: Literal["claim", "subclaim"]
    split: Literal["dev", "test"] = "test"
    
    evaluation_num: int = 100  # only evaluate a part of the dataset because of evaluation cost


def process_gpt_output(gpt_output: dict) -> float:
    """Postprocess the output and get the entailment score {"supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0, "invalid": -1.0}
    The input format is {"prompt": prompt (str), "response": response from gpt (str)}"""
    
    try:
        response: str = gpt_output["response"]
        # get a string between <answer> and </answer>
        answer = response.split("<answer>")[1].split("</answer>")[0]
        return {"supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0}[answer]
    except:
        print("parse failed")
        print(gpt_output["response"])
        return -1.


dataset_dir = Path("../entailment_inputs/oracle_chunks")
oracle_output_dir = Path("../model_outputs/entailment_classification/oracle_chunks")

if __name__ == "__main__":
    args = GPTEvalTap().parse_args()
    
    openai_organization_path = Path("../openai_organization.txt")
    openai_organization = None
    if openai_organization_path.exists():
        openai_organization = openai_organization_path.read_text().strip()
        print(f"Using OpenAI organization: {openai_organization}")
    
    dataset = read_jsonl(dataset_dir / args.claim_subclaim / f"{args.split}.jsonl")
    
    all_article_ids = list(dict.fromkeys([row["meta"]["id"] for row in dataset])) # unique ids but keep order
    evaluated_article_ids = all_article_ids[:args.evaluation_num]
    
    prediction: dict[str, list] = {}
    raw_outputs_list: list[dict] = []
    processed_outputs: dict[str, list] = {}  # key: article_id, value: list of entailment scores
    for row in tqdm(dataset):
        article_id = row["meta"]["id"]
        
        # only evaluate a part of the dataset because of evaluation cost
        if article_id not in evaluated_article_ids:
            break
        
        prompt = get_gpt_prompt(claim=row["claim"], evidence_list=row["evidence"], line_idx=row["meta"]["chunk_idx"])
        output = llm_api(model_name=args.model, prompt=prompt, openai_organization=openai_organization)
        
        raw_outputs_list.append(output)
        processed_outputs.setdefault(article_id, []).append(process_gpt_output(output))
    
    # take the maximum entaiment score as the final output
    final_output = {key: max(values) for key, values in processed_outputs.items()}
    
    # save results
    output_dir = oracle_output_dir / args.claim_subclaim / f"retrieval=oracle,entailment={args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dump_json(final_output, output_dir / f"{args.split}.entailment_score.json")
    dump_jsonl(raw_outputs_list, output_dir / f"{args.split}.raw_outputs.jsonl")
    
