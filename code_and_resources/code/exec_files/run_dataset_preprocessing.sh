for SPLIT in "dev" "test"
    do
    for CLAIM_TYPE in "claim" "subclaim"
        do
        python preprocess_dataset/preprocess_dataset.py --split $SPLIT --claim_type $CLAIM_TYPE
        done
    done
