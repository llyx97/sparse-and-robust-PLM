# A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models

This repository contains implementation of the [paper](https://arxiv.org/abs/2204.11218) "A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models" (accepted by NeruIPS 2022).

## Overview
The main topic of this paper is to investigate **whether there exist PLM subnetworks that are both sparse and robust against dataset bias?**

We call such subnetworks SRNets and explore their existence under different pruning and fine-tuning paradigms, which are illustrated in Figure1.

![](./figures/prune-finetune-paradigms.jpg)


## Requirements
Python3 <br />
torch>1.4.0 <br />


## Prepare Data and Pre-trained Language Models
MNLI and QQP are datasets from the [GLUE](https://gluebenchmark.com/) benchmark. For FEVER, we use the processed [training](https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl) and [evaluation](https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl) data provided by the authors of [FEVER-Symmetric](https://github.com/TalSchuster/FeverSymmetric).

The OOD datasets can be accessed from: [HANS](https://github.com/tommccoy1/hans), [PAWS](https://github.com/google-research-datasets/paws) and [FEVER-Symmetric](https://github.com/TalSchuster/FeverSymmetric).

`bert-base-uncased`, `bert-large-uncased` and `roberta-base` can be downloaded from [huggingface models](https://huggingface.co/models).
