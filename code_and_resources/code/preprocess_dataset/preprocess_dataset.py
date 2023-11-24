from typing import Literal
from pathlib import Path
from copy import deepcopy
import json
from collections import Counter
import random
random.seed(8888)
import warnings
warnings.simplefilter('always', UserWarning)

from tap import Tap
from tqdm import tqdm
import numpy as np
from easy_io import read_jsonl, dump_jsonl

from utils.type_alias import ThreeLabels, RawData, ProcessedData, Chunks


class PreprocessTap(Tap):
    split: str
    claim_type: Literal["claim", "subclaim"]
    word_num_in_chunk: int = 256
    
    add_claim_context: bool = False
    add_evidence_context: bool = False
    
    dataset_dir: Path = Path("../../data/entailment_retrieval/")
    output_dir: Path = Path("../entailment_inputs/")
    
    def process_args(self):
        # process provided path in string type
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)


def split_into_chunks(article_sentences: list[str], article_indices: list[int], word_num_in_chunk: int) -> Chunks:
    """Split article (list of sentences) into overlapping chunks. This function does not split in the middle of sentences.

    Args:
        article_sentences (list[str]): list of sentences in an article
        word_num_in_chunk (int): maximum number of words in each chunk

    Returns:
        Chunks:
    """
    
    chunks_list: list[str] = []
    sentence_idx_list_of_dict: list[dict[Literal["start", "end"], int]] = [{"start": 0}]
    sentence_idx: int = 0
    cur_chunk: list[str] = []
    while True:
        cur_chunk.append(article_sentences[sentence_idx])
        
        # the last sentence in the article
        if sentence_idx == len(article_sentences) - 1:
            if len(cur_chunk) > 0:
                chunks_list.append(cur_chunk)
                sentence_idx_list_of_dict[-1]["end"] = sentence_idx
            else:
                sentence_idx_list_of_dict = sentence_idx_list_of_dict[:-1]
            break
        
        # increment
        sentence_idx += 1
        
        # check length
        cur_chunk_len = np.sum([len(sent.split()) for sent in cur_chunk])
        if cur_chunk_len >= word_num_in_chunk:  # this should be > (mistake) but we will keep using >= for reproducibility
            # remove the last sentence to make the chunk length < word_num_in_chunk
            # however, if the chunk only contains one sentence, we keep it
            if len(cur_chunk) == 1:
                warnings.warn(f"chunk is longer than provided limit:\n{cur_chunk[0]}")
                sentence_idx_list_of_dict[-1]["end"] = sentence_idx - 1
            else:
                cur_chunk = cur_chunk[:-1]
                sentence_idx_list_of_dict[-1]["end"] = sentence_idx - 2  # removed one sentence
            
            chunks_list.append(cur_chunk)
            
            # next chunk start idx, chunks should be overlapped
            sentence_idx -= (len(cur_chunk) - 1) // 2
            
            cur_chunk = []
            sentence_idx_list_of_dict.append({"start": sentence_idx})
    
    sentence_idx_list = [
        [article_indices[idx] for idx in range(sentence_idx_dict["start"], sentence_idx_dict["end"]+1, 1)]
        for sentence_idx_dict in sentence_idx_list_of_dict
    ]
    
    for ck, idx_list in zip(chunks_list, sentence_idx_list):
        assert len(ck) == len(idx_list), f"{len(ck)} != {len(idx_list)}"
    
    return Chunks(
        chunks_list=chunks_list, sentence_idx_list=sentence_idx_list
    )


def get_chunk_label(article_label: ThreeLabels, supporting_sentences_list: list[list[int]],
                    chunk_idx_list: list[int]) -> ThreeLabels:
    """Get chunk label based on the article label and supporting sentences."""
    
    if article_label == "not_supported":
        return "not_supported"
    
    partially_supported: bool = False
    for supporting_sentences in supporting_sentences_list:
        product_set = set(supporting_sentences).intersection(set(chunk_idx_list))
        
        # if article is "supported" and all supporting sentences are included in this chunk
        if article_label == "supported" and len(product_set) == len(supporting_sentences):
            return "supported"

        # if there is at least one supporting sentence in this chunk, we consider it as "partially_supported".
        # keep checking other supporting sentences because this chunk can include all supporting sentences annotated by other annotators.
        if len(product_set) > 0:
            partially_supported = True
    
    if partially_supported:
        return "partially_supported"
    else:
        return "not_supported"


def get_chunk_stats(chunks_list: list[ProcessedData]):
    # count the number of labels
    labels_list: list[str] = []
    for d in chunks_list:
        labels_list.append(d["label"])
    counter = Counter(labels_list)
    
    # output
    output_dict: dict[str, int] = {}
    for label_name in ["supported", "partially_supported", "not_supported"]:
        output_dict[label_name] = counter[label_name]
    
    return output_dict


def get_balanced_output_list_for_evidence_context_data(output_list: list[ProcessedData]):
    """If args.add_evidence_context, we only include sentences. Therefore, there will be too many non-supported cases.
    We make a balanced dataset by randomly sample (discard) non-supported cases.
    """
    
    label_split = {"supported": [], "partially_supported": [], "not_supported": []}
    for d in output_list:
        label_split[d["label"]].append(d)
    
    new_output_list: list[dict] = []
    for label in ["supported", "partially_supported"]:
        new_output_list.extend(label_split[label])
    
    random.shuffle(label_split["not_supported"])
    new_output_list.extend(label_split["not_supported"][:len(label_split["partially_supported"])])
    
    return new_output_list


def check_oracle_chunk(chunk_idx: list[int], oracle_idx: list[int]) -> bool:
    """chunk_idx is a list of sentence indices in a chunk that will be used as the oracle set. oracle_idx is the ground truth oracle idx.
    
    This function check the following property: 
    If len(chunk_idx) >= len(oracle_idx), all elements in oracle_idx should be included in chunk_idx.
    If len(chunk_idx) < len(oracle_idx), all elements in chunk_idx should be included in oracle_idx."""
    
    if len(chunk_idx) >= len(oracle_idx):
        assert set(oracle_idx).issubset(set(chunk_idx)), f"{set(oracle_idx)} is not subset of {set(chunk_idx)}"
    else:
        assert set(chunk_idx).issubset(set(oracle_idx)), f"{set(chunk_idx)} is not subset of {set(oracle_idx)}"


if __name__ == "__main__":
    args = PreprocessTap().parse_args()
    output_dir = args.output_dir

    original_dataset: list[RawData] = read_jsonl(args.dataset_dir / args.claim_type / f"{args.split}.jsonl")
    
    # In this code, we split evidence article into chunks and sentences.
    # chunks are overlapping paragraphs (less than 256 words in default) which are mainly used for entailment classification.
    chunks_output_list: list[dict] = []
    # sentences are mainly used for retrieval.
    sentences_output_list: list[dict] = []
    
    # finally, we get oracle chunks which include all supporting sentences annotated by human annotators.
    # because there are multiple annotators, there are multiple different sets of oracle supporting sentences for each claim/sub-claim.
    # we use this oracle data for training and evaluation with oracle retrieval.
    oracles_output_list: list[dict] = []

    for raw_data in tqdm(original_dataset):
        metadata = raw_data["meta"]
        metadata.pop("claim_context")
        output_dict: ProcessedData = {"label":"", "claim": "", "evidence": "", "meta": metadata}
        
        if args.add_claim_context:
            # add claim context (title of the wikipedia article, section title, and last 128 tokens of the context).
            # this setting does not work well in our experiments, so we did not show this setting in the paper
            output_dict["claim"] = "[context]\n" + raw_data["meta"]["claim_title"] + "\n" + raw_data["meta"]["claim_section"] + \
                "\n" + " ".join(raw_data["meta"]["claim_context"].split()[-128:]) + "\n[claim]\n" + raw_data["claim"]
        else:
            output_dict["claim"] = raw_data["claim"]
        
        # sentences
        for sent_id, evidence_sent in enumerate(raw_data["evidence"]):
            sentence_output_dict = deepcopy(output_dict)
            
            if args.add_evidence_context:
                # add last 128 tokens of the context (previous sentences)
                if sent_id > 0:
                    sentence_output_dict["evidence"] += "[context]\n"
                    # sentence_output_dict["evidence"] += "d["evidence"][0] + "\n"  # title  # we don't use title because it may not be useful for retrieval
                    sentence_output_dict["evidence"] += " ".join(" ".join(raw_data["evidence"][1:sent_id]).split()[-128:])  # 128 tokens
                sentence_output_dict["evidence"] += "\n[evidence]\n"
            
            sentence_output_dict["evidence"] += evidence_sent
            
            sentence_output_dict["label"] = get_chunk_label(raw_data["label"], supporting_sentences_list=raw_data["supporting_sentences"],
                                                            chunk_idx_list=[sent_id])
            
            sentence_output_dict["meta"]["chunk_idx"] = [sent_id]
            sentences_output_list.append(sentence_output_dict)

        # chunks
        # if args.add_evidence_context, we do not use chunks (only sentences) because chunks become too long for T5.
        # args.add_evidence_context is used for retrieval, so we do not need chunks.
        if not args.add_evidence_context:
            chunks = split_into_chunks(raw_data["evidence"], article_indices=list(range(len(raw_data["evidence"]))), word_num_in_chunk=args.word_num_in_chunk)
            for chunk_id, cnk in enumerate(chunks["chunks_list"]):
                chunks_output_dict = deepcopy(output_dict)
                chunks_output_dict["evidence"] = cnk
                chunk_idx_list = chunks["sentence_idx_list"][chunk_id]
                
                chunks_output_dict["label"] = get_chunk_label(
                    raw_data["label"], supporting_sentences_list=raw_data["supporting_sentences"],
                    chunk_idx_list=chunk_idx_list
                )
                
                chunks_output_dict["meta"]["chunk_idx"] = chunk_idx_list
                chunks_output_list.append(chunks_output_dict)
                
                # for non_supported cases, we use chunks as oracle chunks because there is no ground truth supporting sentences.
                # note that non_supported cases can have supporting sentences, but they are for contradictory claims.
                if raw_data["label"] == "not_supported":
                    oracles_output_list.append(chunks_output_dict)
            
            # oracle sentences
            if raw_data["label"] != "not_supported":
                num_oracle_chunks_for_this_case: int = 0
                
                # the number of sets of supporting sentences (supporting sentences by different annotators) can be different for each data.
                # by random sampling, we make at least 3 oracle chunks for each data (this does not apply for some corner cases)
                # this is for avoiding biases caused by the number of oracle chunks (e.g. taking maximum entailment score over all oracle chunks)
                while num_oracle_chunks_for_this_case <= 3:
                    for supporting_sentences in raw_data["supporting_sentences"]:
                        # to avoid leakage (biases) caused by the length of oracle chunks, we use "split_into_chunks" function
                        
                        # however, supporting_sentences may include small number of sentences.
                        # first, we randomly add sentences around the ground truth sentences to "extended_oracle_indices"
                        # the sentences in a chunk will be sorted later.
                        candidate_sentences_added_to_oracle_ = list(
                            set(range(max(supporting_sentences[0]-15, 0), min(supporting_sentences[-1]+25, len(raw_data["evidence"])), 1))
                            - set(supporting_sentences)
                        )
                        candidate_sentences_added_to_oracle = random.Random(num_oracle_chunks_for_this_case).sample(  # shuffle
                            candidate_sentences_added_to_oracle_, len(candidate_sentences_added_to_oracle_)
                        )
                        
                        extended_oracle_indices = supporting_sentences + candidate_sentences_added_to_oracle
                        extended_oracle_chunk_list = [raw_data["evidence"][sentence_idx] for sentence_idx in extended_oracle_indices]
                        
                        # then split the chunk by split_into_chunks
                        # the first chunk is the oracle chunk we use
                        split_oracle_chunk = split_into_chunks(article_sentences=extended_oracle_chunk_list, article_indices=extended_oracle_indices,
                                                            word_num_in_chunk=args.word_num_in_chunk)
                        
                        # postprocess oracle
                        original_oracle_evidence = split_oracle_chunk["chunks_list"][0]
                        original_oracle_sentence_idx = split_oracle_chunk["sentence_idx_list"][0]
                        
                        sorted_oracle_evidence = [original_oracle_evidence[idx] for idx in np.argsort(original_oracle_sentence_idx)]
                        sorted_oracle_sentence_idx = sorted(original_oracle_sentence_idx)
                        check_oracle_chunk(chunk_idx=sorted_oracle_sentence_idx, oracle_idx=supporting_sentences)
                        
                        # save oracle chunks
                        chunks_output_dict = deepcopy(output_dict)
                        chunks_output_dict["evidence"] = sorted_oracle_evidence
                        chunks_output_dict["meta"]["chunk_idx"] = sorted_oracle_sentence_idx
                        chunks_output_dict["meta"]["oracle_idx"] = supporting_sentences
                        chunks_output_dict["label"] = get_chunk_label(
                            raw_data["label"], supporting_sentences_list=raw_data["supporting_sentences"],
                            chunk_idx_list=chunks_output_dict["meta"]["chunk_idx"],
                        )
                        
                        oracles_output_list.append(chunks_output_dict)
                        num_oracle_chunks_for_this_case += 1
                        
                        # add oracle sentences to training data
                        if "train" in args.split:
                            chunks_output_list.append(chunks_output_dict)
    
    #
    # sample from oracle to avoid biases and increasing inference time
    oracle_id_to_list: dict[str, list[dict]] = {}
    oracle_id_list: list[str] = []  # to keep order
    for row in oracles_output_list:
        oracle_id_to_list.setdefault(row["meta"]["id"], []).append(row)
        oracle_id_list.append(row["meta"]["id"])
    
    # remove duplicates from oracle_id_list but keep order
    oracle_id_list = list(dict.fromkeys(oracle_id_list))
    
    sample_num: int = 3  # arbitrary selected
    sampled_oracle_list: list[dict] = []
    for oracle_id in oracle_id_list:
        if len(oracle_id_to_list[oracle_id]) <= sample_num:
            sampled_oracle_list.extend(oracle_id_to_list[oracle_id])
        else:
            # get 3 randomly selected dict from oracle_id_to_list[oracle_id]
            sampled_oracle_list.extend(random.Random(oracle_id).sample(oracle_id_to_list[oracle_id], sample_num))
    
    #
    # save
    for evidence_type, output_list in [["chunks", chunks_output_list], ["sentences", sentences_output_list], ["oracle_chunks", sampled_oracle_list]]:
        # we do not use chunks when args.add_evidence_context is True because evidence context is only for "sentences" setting.
        if args.add_evidence_context and evidence_type in ["chunks", "oracle_chunks"]:
            continue
        
        dir_name = evidence_type
        if args.add_evidence_context:
            dir_name = "evidence_context_" + dir_name
        if args.add_claim_context:
            dir_name = "claim_context_" + dir_name
        
        # if args.add_evidence_context, we only include sentences. Therefore, there will be too many non-supported cases.
        # We make a balanced dataset by randomly sample non-supported cases.
        if args.add_evidence_context and args.split == "train":
            output_list = get_balanced_output_list_for_evidence_context_data(output_list)
        
        output_path: Path = output_dir / dir_name / args.claim_type
        output_path.mkdir(exist_ok=True, parents=True)
        dump_jsonl(output_list, output_path / f"{args.split}.jsonl")
        
        with open(output_path / f"{args.split}_stats.json", "w") as f:
            json.dump(get_chunk_stats(output_list), f, indent=4)
