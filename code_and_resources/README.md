This directory includes the code and model outputs of the experiments in the paper.

## Code
Please refer to [code/README.md](code/README.md) for details.

## Entailment Inputs
Please refer to [code/exec_files/run_dataset_preprocessing.sh](code/exec_files/run_dataset_preprocessing.sh) for how to generate the entailment inputs.

* [entailment_inputs/oracle_chunks](entailment_inputs/oracle_chunks) includes the oracle retrieval dataset. Please refer to [entailment_inputs/oracle_chunks/README.md](entailment_inputs/oracle_chunks/README.md) for details.
* [entailment_inputs/chunks](entailment_inputs/chunks) and [entailment_inputs/sentences](entailment_inputs/sentences) include chunks and sentences from evidence articles.

## Model Outputs
* [model_outputs/entailment_classification/oracle_chunks](model_outputs/entailment_classification/oracle_chunks) includes entailment classification results of the T5 and GPT models on the oracle retrieval dataset.
