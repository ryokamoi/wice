for MODEL in t5-3b-anli-wice t5-3b-anli
do
    for CLAIM_SUBCLAIM in claim subclaim
    do
        for SPLIT in dev test
        do
            python evaluation/postprocess_results.py \
                --entailment_input_jsonl_path ../entailment_inputs/oracle_chunks/${CLAIM_SUBCLAIM}/${SPLIT}.jsonl \
                --prediction_txt_path ../model_outputs/entailment_classification/oracle_chunks/${CLAIM_SUBCLAIM}/retrieval=oracle,entailment=${MODEL}/${SPLIT}.entailment_score.txt \
                --evaluation_num 100  # evaluate only 100 articles because we compare to GPT performance
        done
        python evaluation/evaluate_performance.py \
            --dataset_dev_path ../../data/entailment_retrieval/${CLAIM_SUBCLAIM}/dev.jsonl \
            --dataset_test_path ../../data/entailment_retrieval/${CLAIM_SUBCLAIM}/test.jsonl \
            --prediction_dev_path ../model_outputs/entailment_classification/oracle_chunks/${CLAIM_SUBCLAIM}/retrieval=oracle,entailment=${MODEL}/dev.entailment_score.json \
            --prediction_test_path ../model_outputs/entailment_classification/oracle_chunks/${CLAIM_SUBCLAIM}/retrieval=oracle,entailment=${MODEL}/test.entailment_score.json
    done
done
