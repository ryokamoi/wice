# Code for Reproducing Results in WiCE Paper

Files in [exec_fiels](exec_files) instruct how to run the experiments in this directory.

## Setup

```bash
cd wice/code_and_resources/code
# install and activate conda environment
sh setup.sh
```

## Preprocess Data

This code split articles into chunks and sentences, and also generate the oracle retrieval dataset.

```bash
sh exec_files/run_dataset_preprocessing.sh
```

## GPT Experiments on WiCE

Evaluate GPT-3.5 and GPT-4 on the oracle retrieval dataset.

```bash
# put your OpenAI API key in ../openai_api_key.txt
sh exec_files/evaluate_gpt_on_oracle.sh
```
