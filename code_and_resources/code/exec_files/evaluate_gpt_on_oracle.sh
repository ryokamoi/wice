for MODEL in gpt-3.5-turbo-0613 gpt-4-0613
do
    for CLAIM_SUBCLAIM in claim subclaim
    do
        python gpt_experiments/evaluate_gpt.py --model $MODEL --claim_subclaim $CLAIM_SUBCLAIM --evaluation_num 100

        # using test set for both dataset_dev_path and dataset_test_path is not error
        python evaluation/evaluate_performance.py \
            --dataset_dev_path ../../data/entailment_retrieval/${CLAIM_SUBCLAIM}/test.jsonl \
            --dataset_test_path ../../data/entailment_retrieval/${CLAIM_SUBCLAIM}/test.jsonl \
            --prediction_dev_path ../model_outputs/entailment_classification/oracle_chunks/${CLAIM_SUBCLAIM}/retrieval=oracle,entailment=${MODEL}/test.entailment_score.json \
            --prediction_test_path ../model_outputs/entailment_classification/oracle_chunks/${CLAIM_SUBCLAIM}/retrieval=oracle,entailment=${MODEL}/test.entailment_score.json
    done
done
python evaluation/generate_latex_table.py
