# A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models

This repository contains implementation of the [paper](https://arxiv.org/abs/2204.11218) "A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models" (accepted by NeruIPS 2022).

## Overview
The main topic of this paper is to investigate **whether there exist PLM subnetworks that are both sparse and robust against dataset bias?**

We call such subnetworks **SRNets** and explore their existence under different pruning and fine-tuning paradigms, which are illustrated in Figure1.

![](./figures/prune-finetune-paradigms.jpg)


## Requirements
python3.6 <br />
torch>1.4.0 <br />
install dependencies via `pip install -r requirements.txt`.


## Prepare Data and Pre-trained Language Models
MNLI and QQP are datasets from the [GLUE](https://gluebenchmark.com/) benchmark. For FEVER, we use the processed [training](https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl) and [evaluation](https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl) data provided by the authors of [FEVER-Symmetric](https://github.com/TalSchuster/FeverSymmetric). The OOD datasets can be accessed from: [HANS](https://github.com/tommccoy1/hans), [PAWS](https://github.com/google-research-datasets/paws) and [FEVER-Symmetric](https://github.com/TalSchuster/FeverSymmetric).

By specifying the argument `--model_name_or_path` as `bert-base-uncased`, `bert-large-uncased` or `roberta-base`, the code will automatically download the PLMs. You can also manually download the models from [huggingface models](https://huggingface.co/models) and set `--model_name_or_path` as the path to the model checkpoints.


## Fine-tuning Full BERT

### Fine-tuning with Standard Cross-Entropy Loss
To fine-tune full BERT with standard cross-entropy loss, use the scripts in `scripts/full_bert/std_train`. Taking MNLI as an example, run
```
  bash scripts/full_bert/std_train/mnli.sh
```

### Fine-tuning with Debiasing Loss
The debiasing methods requires bias models' predictions, which are placed in the folder `bias_model_preds`. To fine-tune full BERT with Product-of-Experts(PoE) on MNLI, run
```
  bash scripts/full_bert/robust_train/poe/mnli.sh
```


## Subnetworks from Fine-tuned BERT

### Subnetworks from Standard Fine-tuned BERT
### Subnetworks from Debiasing Fine-tuned BERT


## BERT Subnetworks Fine-tuned in Isolation
### IMP
### Mask Training


## BERT Subnetworks Without Fine-tuning
### Standard Cross-Entropy as Training Objective
### PoE as Training Objective


## Sparse and Unbiased BERT Subnetworks
### Subnetworks from Fine-tuned BERT
### BERT Subnetworks Fine-tuned in Isolation
### BERT Subnetworks Without Fine-tuning


## Refining the SRNets Searching Process
### The Timing to Start Searching SRNets
### Gradual Sparsity Increase
