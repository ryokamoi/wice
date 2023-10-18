from pathlib import Path
from easy_io import read_json

from gpt_experiments.evaluate_gpt import oracle_output_dir


model_names_list = ["t5-3b-anli", "t5-3b-anli-wice", "gpt-3.5-turbo-0613", "gpt-4-0613"]


if __name__ == "__main__":
    results_dict: dict[str, dict[str, dict]] = {}
    for claim_subclaim in ["claim", "subclaim"]:
        results_dict[claim_subclaim] = {}
        for model_name in model_names_list:
            results_path = oracle_output_dir / claim_subclaim / f"retrieval=oracle,entailment={model_name}" / f"test.entailment_score.evaluation.json"
            if results_path.exists():
                results_dict[claim_subclaim][model_name] = read_json(results_path)
            else:
                results_dict[claim_subclaim][model_name] = {}
    
    latex_rows: list[list[str]] = [["Model", "F1", "Accuracy", "F1", "Accuracy"]]
    for model_name, model_str in zip(model_names_list, ["T5-3B (ANLI)\\phantom{+WiCE}", "T5-3B (ANLI+WiCE)", "GPT-3.5", "GPT-4"]):
        row = [f"{model_str:30s}"]
        for claim_subclaim in ["claim", "subclaim"]:
            for score in ["f1", "accuracy"]:
                if len(results_dict[claim_subclaim][model_name]) == 0:
                    row.append("")
                else:
                    row.append(f"{100*results_dict[claim_subclaim][model_name][score]['0_vs_1,2'][score]:.1f}")
        latex_rows.append(row)
    latex_rows_str = [" & ".join(row) + " \\\\" for row in latex_rows]
    
    # output
    table_path = oracle_output_dir / "result_table.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(latex_rows_str))
