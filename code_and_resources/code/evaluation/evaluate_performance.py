import json
from typing import Literal, Optional
from pathlib import Path
from tqdm import tqdm

from tap import Tap
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
import numpy as np
from easy_io import read_jsonl, read_list_from_text_file

from utils.type_alias import RawData, DevTest


class EvaluationTap(Tap):
    dataset_test_path: str
    dataset_dev_path: str
    prediction_test_path: str
    prediction_dev_path: str
    evaluate_bootstrap: bool = False
    
    filtering: Optional[str] = None  # this is not used in the WiCE paper
    document_level: bool = False  # this is not used in the WiCE paper


label_str_to_int: dict[str, int] = {
    "supported": 0, "partially_supported": 1, "not_supported": 2,
}


def binarize_scores(threshold: float, scores: list[float]) -> list[int]:
    return (np.array(scores) > threshold).astype(int).tolist()


def get_evaluation_scores(score_name: Literal["f1", "accuracy", "balanced_accuracy"], test_labels: list[int], dev_labels: list[int],
                          test_scores: list[float], dev_scores: list[float],
                          provided_threshold: Optional[float]=None,
                          threshold_min: float=0., threshold_max: float=1.) -> dict:
    threshold: float = 0.
    max_score: float = 0.
    score_list: list[float] = []
    thresholds_list: list[float] = []
    if provided_threshold is None:  # use dev set to select the best threshold
        for th in np.arange(threshold_min, threshold_max, step=(threshold_max - threshold_min) / 100):
            func = {
                "f1": f1_score, "accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score,
            }[score_name]
            score = func(y_true=dev_labels, y_pred=binarize_scores(th, dev_scores))
            
            score_list.append(score)
            thresholds_list.append(th)
            if max_score < score:
                threshold = th
                max_score = score
    else:
        threshold = provided_threshold
    
    # test
    output_dict = {}
    if score_name == "f1":
        prfs = precision_recall_fscore_support(y_true=test_labels, y_pred=binarize_scores(threshold, test_scores), average="binary")
        output_dict.update({"f1": prfs[2], "precision": prfs[0], "recall": prfs[1]})
    elif score_name == "accuracy":
        accuracy = accuracy_score(y_true=test_labels, y_pred=binarize_scores(threshold, test_scores))
        output_dict.update({"accuracy": accuracy})
    elif score_name == "balanced_accuracy":
        ba = balanced_accuracy_score(y_true=test_labels, y_pred=binarize_scores(threshold, test_scores))
        output_dict.update({"balanced_accuracy": ba})
    
    output_dict["threshold"] = threshold
    
    output_dict["dev"] = {
        "scores": score_list, "thresholds": thresholds_list,
    }

    return output_dict


def filter_prediction(datasets: dict[DevTest, list[RawData]], predictions: dict[DevTest, dict[str, dict]],
                      filtering_type: Optional[str]=None) -> tuple[dict[DevTest, list[RawData]], dict[DevTest, dict[str, dict]]]:
    """This filtering is not used in the final version of WiCE paper."""
    
    if filtering_type is None:
        return datasets, predictions
    
    filtered_dataset: dict[DevTest, list[RawData]] = {}
    filtered_predictions: dict[DevTest, dict[str, dict]] = {}
    for split in ["dev", "test"]:
        filtered_dataset[split] = []
        filtered_predictions[split] = {}
        for d in datasets[split]:
            use_this_case = False
            if d["label"] == "not_supported":
                use_this_case = True
            else:
                if filtering_type == "multiple_supporting_sentences":
                    if max([len(s) for s in d["supporting_sentences"]]) >= 2:
                        use_this_case = True
                elif filtering_type == "distant":
                    min_dist = 100
                    for s in d["supporting_sentences"]:
                        min_dist = min(min_dist, max(s) - min(s))
                    if min_dist >= 5:
                        use_this_case = True
                else:
                    raise ValueError(f"{filtering_type} is not a valid value of filtering_type")
            
            if use_this_case:
                filtered_dataset[split].append(d)
                article_id = d["meta"]["id"]
                filtered_predictions[split][article_id] = predictions[split][article_id]            
    
    return filtered_dataset, filtered_predictions


def convert_to_document_level(predictions: dict[str, float]):
    output_dict: dict[str, float] = {}
    for sentence_id, score in predictions.items():
        article_id = "_".join(sentence_id.split("_")[:-1])  # remove sentence or chunk id
        if article_id in output_dict.keys():
            output_dict[article_id] = min(output_dict[article_id], score)
        else:
            output_dict[article_id] = score
    
    return output_dict


def make_prediction_list(datasets: dict[DevTest, list[RawData]], predictions: dict[DevTest, dict[str, dict]]) -> tuple[dict[DevTest, list[int]], dict[DevTest, list[float]]]:
    """Convert dataset and predictions to list of labels and scores."""
    
    scores_dict_of_list: dict[DevTest, list[float]] = {}
    labels_dict_of_list: dict[DevTest, list[int]] = {}
    for split in ["dev", "test"]:
        scores_dict_of_list[split] = []
        labels_dict_of_list[split] = []

        for d in datasets[split]:
            if d["meta"]["id"] in predictions[split].keys():
                score = predictions[split][d["meta"]["id"]]
                scores_dict_of_list[split].append(score)
                labels_dict_of_list[split].append(label_str_to_int[d["label"]])
    
    return labels_dict_of_list, scores_dict_of_list


def get_list_of_prediction_scores_for_target_labels(
        target_labels: list[int], label_list: list[int], score_list: list[float]
    ) -> list[float]:
    """Get list of prediction scores for target labels."""
    output_list: list[float] = []
    
    for idx, score in enumerate(score_list):
        if label_list[idx] in target_labels:
            output_list.append(score)
    
    return output_list


def evaluate_entailment_classification(labels_dict_of_list: dict[DevTest, list[int]], scores_dict_of_list: dict[DevTest, list[float]], provided_thresholds_dict: Optional[dict]=None):
    evaluation_output: dict = {"label_distribution": {}, "roc": {}, "f1": {}, "accuracy": {}, "balanced_accuracy": {}}
    for label in [0, 1, 2]:
        evaluation_output["label_distribution"][label] = np.sum(np.array(labels_dict_of_list["test"]) == label).item()
    
    # evaluate classification performance between each pair of labels
    # [[0], [1, 2]]] means the classification performance between supported vs. partially_supported + not_supported (supported or not)
    for x, y in [[[0], [1]], [[1], [2]], [[0], [2]], [[0], [1, 2]]]:
        y_true_dict: dict[DevTest, list[int]] = {}
        y_score_dict: dict[DevTest, list[float]] = {}
        for split in ["dev", "test"]:
            prediction_scores_for_data_with_target_labels = []
            for target_labels in [x, y]:
                prediction_scores_for_data_with_target_labels.append(
                    get_list_of_prediction_scores_for_target_labels(
                        target_labels=target_labels, label_list=labels_dict_of_list[split], score_list=scores_dict_of_list[split]
                    )
                )
            
            y_true_dict[split] = []
            for binary in [0, 1]:
                y_true_dict[split] += [[1, 0][binary]] * len(prediction_scores_for_data_with_target_labels[binary])  # 1 for entailed!!
            y_score_dict[split] = prediction_scores_for_data_with_target_labels[0] + prediction_scores_for_data_with_target_labels[1]
        
        label_key = f"{','.join(map(str, x))}_vs_{','.join(map(str, y))}"
        
        # auroc
        fpr, tpr, _ = roc_curve(y_true=y_true_dict["test"], y_score=y_score_dict["test"])
        try:
            auroc = roc_auc_score(y_true=y_true_dict["test"], y_score=y_score_dict["test"])
        except ValueError:
            auroc = -1
        
        evaluation_output["roc"][label_key] = {
            "roc": auroc,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        }
        
        # f1 and accuracy
        for score_name in ["f1", "accuracy", "balanced_accuracy"]:
            if provided_thresholds_dict is not None:
                provided_threshold = provided_thresholds_dict[score_name][label_key]["threshold"]
            else:
                if score_name == "f1":
                    provided_threshold = None
                else:
                    # use threshold for F1 score for accuracy and BA
                    provided_threshold = evaluation_output["f1"][label_key]["threshold"]
            
            # if provided_threshold is None, the best parameter is selected by using the dev set
            evaluation_output[score_name][label_key] = get_evaluation_scores(
                score_name=score_name,
                test_labels=y_true_dict["test"], dev_labels=y_true_dict["dev"],
                test_scores=y_score_dict["test"], dev_scores=y_score_dict["dev"],
                provided_threshold=provided_threshold
            )
    
    return evaluation_output


def get_article_for_bootstrap(dataset: list[dict], article_ids_list: list[str]) -> list[dict]:
    """Get article data for bootstrap."""
    
    dataset_id_to_data: dict[str, dict] = {}
    for d in dataset:
        dataset_id_to_data[d["meta"]["id"]] = d
    
    output_list = []
    for article_id in article_ids_list:
        output_list.append(dataset_id_to_data[article_id])
    
    return output_list


if __name__ == "__main__":
    args = EvaluationTap().parse_args()
    
    datasets: dict[DevTest, list[RawData]] = {}
    for split, path in [["dev", args.dataset_dev_path], ["test", args.dataset_test_path]]:
        datasets[split] = read_jsonl(path)
    
    predictions: dict[DevTest, dict[str, dict]] = {}
    for split, path in [["dev", args.prediction_dev_path], ["test", args.prediction_test_path]]:
        with open(path, "r") as f:
            predictions[split] = json.load(f)
        
        if args.document_level:
            predictions[split] = convert_to_document_level(predictions[split])
    
    # the filtering is not used in the final version of WiCE paper
    datasets, predictions = filter_prediction(datasets, predictions, filtering_type=args.filtering)
    
    # evaluation
    labels_dict_of_list, scores_dict_of_list = make_prediction_list(datasets=datasets, predictions=predictions)
    evaluation_output = evaluate_entailment_classification(labels_dict_of_list=labels_dict_of_list, scores_dict_of_list=scores_dict_of_list)
    
    # save results
    suffix = ".evaluation.json" if args.filtering is None else f".{args.filtering}.evaluation.json"
    with open(Path(args.prediction_test_path).with_suffix(suffix), "w") as f:
        json.dump(evaluation_output, f, indent=4)
    
    # bootstrap
    if args.evaluate_bootstrap:
        metric_names = ["accuracy", "f1", "roc"]
        bootstrap_results: dict[str, list[float]] = {key: [] for key in metric_names}
        
        for bootstrap_id in tqdm(range(1000)):
            bootstraped_dataset: dict[str, list] = {"dev": []}  # do not use dev set
            for split, dataset_path in [["test", args.dataset_test_path]]:
                # bootstrap indices should be generated beforehand
                article_ids_list: list[str] = read_list_from_text_file(
                    Path("bootstrap") / Path(dataset_path).parent.name / f"{Path(dataset_path).stem}_{bootstrap_id:05d}.txt",
                    element_type="str"
                )
                bootstraped_dataset[split] = get_article_for_bootstrap(datasets[split], article_ids_list)

            labels_dict_of_list, scores_dict_of_list = make_prediction_list(datasets=bootstraped_dataset, predictions=predictions)
            
            # provide threshold of the original dev set
            evaluation_output = evaluate_entailment_classification(labels_dict_of_list=labels_dict_of_list, scores_dict_of_list=scores_dict_of_list,
                                               provided_thresholds_dict=evaluation_output)
            
            for metric in metric_names:
                bootstrap_results[metric].append(evaluation_output[metric]["0_vs_1,2"][metric])
        
        with open(Path(args.prediction_test_path).with_suffix(".bootstrap.json"), "w") as f:
            json.dump(bootstrap_results, f, indent=4)
