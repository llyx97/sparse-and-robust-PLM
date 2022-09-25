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
Download the [GLUE](https://gluebenchmark.com/) datasets to `imp_and_fine_tune/glue` and the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) v1.1 dataset to `mask_training/data/squad`.

`bert-base-uncased`, `bert-large-uncased` and `roberta-base` can be downloaded from [huggingface models](https://huggingface.co/models).
