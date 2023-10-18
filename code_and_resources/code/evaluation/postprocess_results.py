import json
from pathlib import Path
from typing import Union

from easy_io import read_jsonl, read_list_from_text_file
from tap import Tap

from utils.type_alias import RawData


class PostprocessTap(Tap):
    entailment_input_jsonl_path: str
    prediction_txt_path: str
    
    evaluation_num: int = 100  # number of articles to evaluate


if __name__ == "__main__":
    args = PostprocessTap().parse_args()
    
    entailment_input_dataset: list[RawData] = read_jsonl(args.entailment_input_jsonl_path)
    nli_prediction: list[float] = [float(s) for s in read_list_from_text_file(args.prediction_txt_path)]
    assert len(entailment_input_dataset) == len(nli_prediction)
    
    maximum_entailment_dict: dict[str, Union[dict[int, float], float]] = {}
    for row_id, article_fact_dict in enumerate(entailment_input_dataset):
        article_idx: str = article_fact_dict["meta"]["id"]
        entailment_score: float = nli_prediction[row_id]
        
        if article_idx not in maximum_entailment_dict.keys():
            # don't add new article if the number of articles is enough
            if len(maximum_entailment_dict.keys()) == args.evaluation_num:
                break
            
            maximum_entailment_dict[article_idx] = entailment_score
        else:
            maximum_entailment_dict[article_idx] = max(maximum_entailment_dict[article_idx], entailment_score)
    
    with open(Path(args.prediction_txt_path).with_suffix(".json"), "w") as f:
        json.dump(maximum_entailment_dict, f, indent=4)
