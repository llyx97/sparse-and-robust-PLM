# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os, sys, json, re, pickle
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch.nn.utils.prune as prune

import numpy as np
import torch

import sys
sys.path.append('transformer/src/')

from hg_transformers.configuration_auto import AutoConfig
from hg_transformers.modeling_auto import AutoModelForSequenceClassification
from hg_transformers.tokenization_auto import AutoTokenizer
from hg_transformers.trainer_utils import EvalPrediction
from load_dataset import GlueDataset
from hg_transformers.data.datasets.glue import GlueDataTrainingArguments
from hg_transformers.hf_argparser import HfArgumentParser
from hg_transformers.training_args import TrainingArguments as BaseTrainingArguments
from hg_transformers.data.processors.glue import glue_output_modes
from hg_transformers.data.processors.glue import glue_tasks_num_labels
from hg_transformers.data.metrics import glue_compute_metrics
from hg_transformers.trainer import set_seed
from hg_transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule
#from optimization import AdamW
from sklearn.metrics import f1_score
from hg_transformers.optimization import AdamW
import hg_transformers



logger = logging.getLogger(__name__)

ood_dataset_names = {"mnli": ['hans'],
                    "qqp": ['paws_qqp', 'paws_wiki'],
                    "fever": ['sym1', 'sym2']}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



def summarize_results(task, output_dir):
    lines = """import re, os
import numpy as np

task = '%s'"""%task.lower() + """
seeds = [i for i in range(1, 4)]
pattern = re.compile(r'-?\d+\.?\d*e?-?\d*?')

scores = []
for seed in seeds:
        filename = os.path.join(str(seed), 'eval_results_%s.txt'%task)
        file = open(filename, 'r')
        lines = file.readlines()
        s = float(pattern.findall(lines[-1])[0])
        print('%d: %.3f'%(seed, s))
        scores.append(s)
        file.close()
score = np.mean(scores)
std = np.std(scores)
print('Avg score: %.3f'%(score))
print('Std: %.3f'%(std))
    """
    if not os.path.exists(output_dir[:-2]+'/summarize_results.py'):
        file = open(os.path.join(output_dir[:-2], 'summarize_results.py'), 'w')
        file.write(lines)
        file.close()


def mag_pruning(model,px):

    print('Start magnitude pruning with zero rate %.2f'%px)
    modules_to_prune =[]
    for ii in range(12):
        modules_to_prune.append('encoder.layer.%d.attention.self.query'%ii)
        modules_to_prune.append('encoder.layer.%d.attention.self.key'%ii)
        modules_to_prune.append('encoder.layer.%d.attention.self.value'%ii)
        modules_to_prune.append('encoder.layer.%d.attention.output.dense'%ii)
        modules_to_prune.append('encoder.layer.%d.intermediate.dense'%ii)
        modules_to_prune.append('encoder.layer.%d.output.dense'%ii)

    modules_to_prune.append('pooler.dense')
    for name, module in model.named_modules():
        if name in modules_to_prune:
            prune.l1_unstructured(module, 'weight', amount=px)
    prune.l1_unstructured(model.embeddings.word_embeddings, 'weight', amount=px)

def see_weight_rate(model, model_type):
    sum_list = 0
    zero_sum = 0
    if 'bert.encoder.layer.0.attention.self.value.weight' in model.state_dict():
        suffix = '.weight'
    else:
        suffix = '.weight_mask'
    for ii in range(model.config.num_hidden_layers):
        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query%s'%suffix].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query%s'%suffix] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key%s'%suffix].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key%s'%suffix] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value%s'%suffix].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value%s'%suffix] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense%s'%suffix].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense%s'%suffix] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense%s'%suffix].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense%s'%suffix] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense%s'%suffix].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense%s'%suffix] == 0))

    sum_list = sum_list+float(model.state_dict()['%s.pooler.dense%s'%(model_type, suffix)].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.pooler.dense%s'%(model_type, suffix)] == 0))
    #sum_list = sum_list+float(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type].nelement())
    #zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type] == 0))
    return 100*zero_sum/sum_list


def pruning_model_with_mask(model, mask_dict, mask_classifier, model_type='bert'):
    parameters_to_prune =[]
    mask_list = []
    suffix = '.weight_mask' if '_mask' in list(mask_dict.keys())[0] else '.weight'
    if model_type=='bert':
        bert_model = model.bert
    elif model_type=='roberta':
        bert_model = model.roberta
    for ii in range(model.config.num_hidden_layers):
        parameters_to_prune.append(bert_model.encoder.layer[ii].attention.self.query)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query%s'%suffix])
        parameters_to_prune.append(bert_model.encoder.layer[ii].attention.self.key)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key%s'%suffix])
        parameters_to_prune.append(bert_model.encoder.layer[ii].attention.self.value)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value%s'%suffix])
        parameters_to_prune.append(bert_model.encoder.layer[ii].attention.output.dense)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense%s'%suffix])
        parameters_to_prune.append(bert_model.encoder.layer[ii].intermediate.dense)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense%s'%suffix])
        parameters_to_prune.append(bert_model.encoder.layer[ii].output.dense)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.output.dense%s'%suffix])

    parameters_to_prune.append(bert_model.pooler.dense)
    mask_list.append(mask_dict['%s.pooler.dense%s'%(model_type, suffix)])
    if 'embedding' in ' '.join(mask_dict):
        parameters_to_prune.append(model.embeddings.word_embeddings)
        mask_list.append(mask_dict['%s.embeddings.word_embeddings%s'%(model_type, suffix)])
    if 'classifier' in ' '.join(mask_dict) and mask_classifier:
        if model_type=='bert':
            parameters_to_prune.append(model.classifier)
            mask_list.append(mask_dict['classifier%s'%suffix])
        elif model_type=='roberta':
            parameters_to_prune.append(model.classifier.dense)
            parameters_to_prune.append(model.classifier.out_proj)
            mask_list.append(mask_dict['classifier.dense%s'%suffix])
            mask_list.append(mask_dict['classifier.out_proj%s'%suffix])

    for ii in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list[ii].bool())


def prune_with_mask(model, mask_dir, component_type):
    """
    component_type = 'head' or 'ffn'
    """
    logger.info("Loading mask from %s"%mask_dir)
    mask = torch.from_numpy(np.load(mask_dir))
    to_prune = {}
    for layer in range(len(mask)):
        to_mask = [h[0] for h in (1 - mask[layer].long()).nonzero().tolist()]
        to_prune[layer] = to_mask
    assert sum(len(h) for h in to_prune.values()) == (1 - mask.long()).sum().item()

    logger.info("%s zero rate:%.3f"%(component_type, (mask==0).view(-1).sum().div(float(mask.numel()))))
    if component_type=='head':
        logger.info(f"Pruning heads {to_prune}")
        model.prune_heads(to_prune)
    elif component_type=='ffn':
        model.prune_ffns(to_prune)


def load_mask_and_prune(mask_dir, model, model_args):
    logger.info('Loading mask from %s'%mask_dir)
    if 'mask.pt' in os.listdir(mask_dir):
        mask = torch.load(os.path.join(mask_dir, 'mask.pt'))
    elif 'pytorch_model.bin' in os.listdir(mask_dir):
        mask, weights = {}, {}
        model_dict = torch.load(os.path.join(mask_dir, 'pytorch_model.bin'))
        for key in model_dict.keys():
            if 'mask' in key:
                mask[key] = model_dict[key].cpu().bool()
            elif 'orig' in key:
                weights[key.replace('_orig', '')] = model_dict[key].cpu()
        logger.info('Loading weights from pruned model')
        model_dict = model.state_dict()
        model_dict.update(weights)
        model.load_state_dict(model_dict)
    else:
        raise FileNotFoundError('No mask file found in %s'%mask_dir)

    print('Mask for %s.encoder.layer.1.attention.self.query.weight: \n'%model_args.model_type)
    try:
        print(mask['%s.encoder.layer.1.attention.self.query.weight_mask'%model_args.model_type])
    except KeyError:
        print(mask['%s.encoder.layer.1.attention.self.query.weight'%model_args.model_type])
    pruning_model_with_mask(model, mask, model_args.mask_classifier, model_args.model_type)
    return model


@dataclass
class TrainingArguments(BaseTrainingArguments):
    """
    This is a subclass of transformers.TrainingArguments
    """
    weight_rewind: str2bool = field(
        default=False, metadata={"help": "Whether to rewind weights in IMP."}
    )
    best_metric: str = field(
        default='eval_acc', metadata={"help": "The evaluation metric for best checkpoint selection"}
    )
    prune_global_imp: str2bool = field(
        default=True, metadata={"help": "Whether to conduct global pruning in IMP."}
    )
    save_imp_model: str2bool = field(
        default=False, metadata={"help": "Whether to save the imp models."}
    )
    save_best_model: str2bool = field(
        default=True, metadata={"help": "Whether to save the best model checkpoint."}
    )
    save_final_model: str2bool = field(
        default=False, metadata={"help": "Whether to save the model at the end of training."}
    )
    train_subset_size: Optional[int] = field(
        default=0, metadata={"help": "The number of data in the subset for training. If equals to 0, use the entire training set."}
    )
    robust_training: str = field(
        default=None, metadata={"help": "The evaluation metric for best checkpoint selection",
            "choices": [None, "reweighting", "regularization", "poe"]}
    )
    global_grad_clip: str2bool = field(
        default=True, metadata={"help": "Whether to conduct grad clip globally."}
    )
    bias_dir: str = field(
        default=None, metadata={"help": "The directorty of the bias degree file."}
    )
    anneal_bias_range: Optional[str] = field(
        default=None, metadata={"help": "The range of bias degree annealing, separated by _."}
    )
    teacher_prob_dir: str = field(
        default=None, metadata={"help": "The directorty of teacher model's predicted probability file."}
    )

@dataclass
class DataTrainingArguments(GlueDataTrainingArguments):
    """
    This is a subclass of transformers.GlueDataTrainingArguments
    """
    synthetic_data: str2bool = field(
        default=False, metadata={"help": "Whether to train and test with synthetic bias data."}
    )
    eval_ood: str2bool = field(
        default=True, metadata={"help": "Whether to evaluate with ood dataset."}
    )
    synthetic_rate: Optional[float] = field(
        default=0.3, metadata={"help": "The percentage of training examples to add synthetic feature"}
    )
    bias_rate: Optional[float] = field(
        default=0.9, metadata={"help": "The percentage of training examples to add synthetic feature"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_type: str = field(
        metadata={"help": "Type of the model"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    zero_rate: Optional[float] = field(
        default=0., metadata={"help": "The percentage of 0 in model weights."}
    )
    is_prune: str2bool = field(
        default=False, metadata={"help": "Whether to perform pruning."}
    )
    is_imp: str2bool = field(
        default=False, metadata={"help": "Whether to perform imp."}
    )
    root_dir: Optional[str] = field(
        default=None, metadata={"help": "The root directory."}
    )
    mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The usntructured mask directory."}
    )
    load_classifier: str2bool = field(
        default=False, metadata={"help": "Whether to load the classifier, if it is trained together with the masks."}
    )
    mask_classifier: str2bool = field(
        default=True, metadata={"help": "Whether to mask the classifier weights during re-training."}
    )
    ffn_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The ffn mask directory."}
    )
    head_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The head mask directory."}
    )
    mask_seed: Optional[int] = field(
        default=1, metadata={"help": "The seed for random masking."}
    )
    prune_ffn: str2bool = field(
        default=False, metadata={"help": "Whether to prune FFNs."}
    )
    prune_head: str2bool = field(
        default=False, metadata={"help": "Whether to prune attention heads."}
    )
    structured: str2bool = field(
        default=False, metadata={"help": "Whether to perform structured masking."}
    )
    struc_prun_type: Optional[str] = field(
            default="one_step", metadata={"help": "One step or iterative pruning.", "choices":["one_step", "iterative", "random"]}
    )


def rebuild_trainset(train_dataset, bias_dir, teacher_prob_dir=None):
    #biases = np.load(os.path.join(bias_dir, 'log_probs.npy'))
    if '.npy' in bias_dir:
        biases = np.load(bias_dir)
    elif '.pkl' in bias_dir:
        biases = pickle.load(open(bias_dir, 'rb'))
    elif '.json' in bias_dir:
        biases = json.load(open(bias_dir, 'r'))

    features = train_dataset.features
    new_features = []

    if isinstance(biases, dict):
        biases = {int(k): biases[k] for k in biases}
    else:
        assert len(biases)==len(features)

    if teacher_prob_dir is not None:
        teacher_probs = np.load(os.path.join(teacher_prob_dir, 'probs.npy'))
    else:
        teacher_probs = None

    logger.info("original len: {}".format(str(len(features))))
    pattern = re.compile(r'-?\d+\.?\d*e?-?\d*?')
    for i, fe in enumerate(features):
        # biases are indxed by the id column, while teacher_probs are indexed by the line number
        ind = abs(int(pattern.findall(fe.example_id)[0]))
        if isinstance(biases, dict) and not ind in biases:
            continue
        fe.bias = biases[ind]
        fe.teacher_probs = teacher_probs[i] if teacher_probs is not None else None
        new_features.append(fe)
    logger.info("filtered len: {}".format(str(len(new_features))))

    train_dataset.features = new_features
    return train_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_path = model_args.model_name_or_path if model_args.config_name is None else model_args.config_name
    config = AutoConfig.from_pretrained(
        config_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    tokenizer_path = model_args.model_name_or_path if model_args.tokenizer_name is None else model_args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        #model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='train') if training_args.do_train else None
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='dev') if training_args.do_eval else None
    test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='test') if training_args.do_predict else None

    ood_datasets = None
    if data_args.eval_ood:
        if data_args.task_name.lower() == 'mnli':
            ood_datasets = [GlueDataset(data_args, tokenizer=tokenizer, dataset_name='hans', mode='test')]
        elif data_args.task_name.lower() == 'qqp':
            ood_datasets = [GlueDataset(data_args, tokenizer=tokenizer, dataset_name='paws_qqp', mode='dev')] \
                    + [GlueDataset(data_args, tokenizer=tokenizer, dataset_name='paws_wiki', mode='test')]
        elif data_args.task_name.lower() == 'fever':
            ood_datasets = [GlueDataset(data_args, tokenizer=tokenizer, dataset_name='sym1', mode='test')] \
                    + [GlueDataset(data_args, tokenizer=tokenizer, dataset_name='sym2', mode='test')]

    # Build the training set with bias degree of each example for robust training
    if training_args.robust_training is not None and train_dataset is not None:
        logger.info("Building the training set with bias degree")
        assert training_args.bias_dir is not None, "Please provide a file of the bias degree"
        train_dataset = rebuild_trainset(train_dataset, training_args.bias_dir, training_args.teacher_prob_dir)
        from load_dataset import DataCollatorWithBias
        data_collator = DataCollatorWithBias()
    else:
        data_collator = None

    # Select a subset for training
    if training_args.train_subset_size > 0:
        train_dataset.features = np.random.choice(train_dataset.features, size=training_args.train_subset_size, replace=False)

    if model_args.zero_rate > 0 or model_args.is_prune:
#        mask_seed = model_args.mask_seed if 'rand' in model_args.struc_prun_type else ''
        if model_args.structured:
            #head_mask, ffn_mask = torch.from_numpy(np.load(model_args.head_mask_dir)), torch.from_numpy(np.load(model_args.ffn_mask_dir))
            #heads_to_prune, ffns_to_prune = {}, {}
            #for layer in range(len(head_mask)):
            #    heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
            #    heads_to_prune[layer] = heads_to_mask
            #    ffns_to_mask = [f[0] for f in (1 - ffn_mask[layer].long()).nonzero().tolist()]
            #    ffns_to_prune[layer] = ffns_to_mask
            #assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
            #assert sum(len(f) for f in ffns_to_prune.values()) == (1 - ffn_mask.long()).sum().item()

            #logger.info(f"Pruning heads {heads_to_prune}")
            #logger.info("Head zero rate:%.3f"%float((head_mask==0).view(-1).sum().true_divide(head_mask.numel())))
            #logger.info("FFN zero rate:%.3f"%float((ffn_mask==0).view(-1).sum().true_divide(ffn_mask.numel())))
            #model.prune_heads(heads_to_prune)
            #model.prune_ffns(ffns_to_prune)

            if model_args.prune_head:
                prune_with_mask(model, model_args.head_mask_dir, 'head')
            if model_args.prune_ffn:
                prune_with_mask(model, model_args.ffn_mask_dir, 'ffn')
        else:
            if model_args.mask_dir is not None:
                assert not model_args.is_imp, "Set mask_dir to None when using IMP!"
                mask_dir = model_args.mask_dir
                model = load_mask_and_prune(mask_dir, model, model_args)
                if model_args.load_classifier:
                    model.classifier = torch.load(os.path.join(model_args.mask_dir, 'classifier.bin')).to(training_args.device)

            zero = see_weight_rate(model, model_args.model_type)
            print('model 0:',zero)

    param_count = 0
    for n, p in model.named_parameters():
        param_count += p.nelement()
    param_count /= 1e6

    def compute_metrics_ood(p: EvalPrediction, dataset_name=None) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
            if data_args.task_name=='mnli':
                preds[preds == 2] = 0
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        metrics = glue_compute_metrics(data_args.task_name, preds, p.label_ids)
        # In paws-qqp and paws-wiki, the pos/neg classes are imbalanced
        if data_args.task_name=='qqp':
            metrics['duplicate_acc'] = (preds[p.label_ids==1].sum()*1.) / (p.label_ids.sum()*1.)
            metrics['non-duplicate_acc'] = ((preds[p.label_ids==0]==0).sum()*1.) / ((p.label_ids==0).sum()*1.)
            metrics['non-duplicate_f1'] = f1_score(y_true=p.label_ids, y_pred=preds, pos_label=0)
            metrics['average_f1'] = f1_score(y_true=p.label_ids, y_pred=preds, pos_label=0, average='weighted')
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"{dataset_name}_eval_{key}"] = metrics.pop(key)
        return metrics

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    def write_ood_preds(preds, label_list, dataset_name):
        ood_pred_file = os.path.join(training_args.output_dir, '%s_preds.txt'%dataset_name)
        preds = np.argmax(preds, axis=1)
        preds_ = [label_list[p] for p in preds]
        with open(ood_pred_file, "w") as writer:
            logger.info("***** Writing OOD Predictions *****")
            writer.write("pairID,gold_label\n")
            for i, pred in enumerate(preds_):
                writer.write("ex%d,%s\n"%(i, pred))


    # Initialize our Trainer
    if model_args.is_prune and model_args.is_imp and not model_args.mask_dir:
        from hg_transformers.trainer_imp import Trainer
    else:
        from hg_transformers.trainer import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ood_datasets=ood_datasets,
        compute_metrics=compute_metrics,
        compute_metrics_ood=compute_metrics_ood,
        optimizers=None,
        data_collator=data_collator,
    )

    fw_args = open(training_args.output_dir + '/args.txt', 'w')
    fw_args.write(str(training_args)+'\n\n')
    fw_args.write(str(model_args)+'\n\n')
    fw_args.write(str(data_args)+'\n\n')
    fw_args.write("Model size:%.2fM"%param_count+'\n\n')
    fw_args.close()

    # Training
    if training_args.do_train:
        _, best_score, results_at_best_score = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
        if training_args.save_final_model:
            final_model_path = os.path.join(training_args.output_dir, 'final_model')
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)

        output_eval_file = os.path.join(
                training_args.output_dir, f"best_eval_results_{eval_dataset.args.task_name}.txt"
            )
        if results_at_best_score is not None:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Best Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in results_at_best_score.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
    #zero = see_weight_rate(model)
    #print('model 0:',zero)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        #summarize_results(data_args.task_name, training_args.output_dir)
        if training_args.do_train and results_at_best_score is not None and training_args.save_best_model:
            logger.info("*** Loading best checkpoint ***")
            model_dict = torch.load(os.path.join(training_args.output_dir, 'pytorch_model.bin'))
            trainer.model.load_state_dict(model_dict)

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = {data_args.task_name: eval_dataset}
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.update({'mnli-mm': GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode='dev')})
        if ood_datasets is not None:
            for od in ood_datasets:
               eval_datasets.update({od.dataset_name: od})

        for task_name, eval_dataset in eval_datasets.items():
            if task_name not in ood_dataset_names[data_args.task_name]:
                # IID dev set
                result = trainer.evaluate(eval_dataset=eval_dataset)
            else:
                # OOD test set
                result, preds = trainer.evaluate_ood(eval_dataset, compute_metrics_ood)
                write_ood_preds(preds, eval_datasets[data_args.task_name].get_labels(), eval_dataset.dataset_name)
            results.update(result)

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results*****")
            for key, value in results.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    if training_args.bias_dir is None and training_args.teacher_prob_dir is None:
        logger.info("*** Compute the predicted logits of the training set ***")
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='train')
        eval_dataloader = trainer.get_eval_dataloader(train_dataset)
        output = trainer._prediction_loop(eval_dataloader, description="Evaluation")
        logits = torch.tensor(output.predictions)
        probs = torch.nn.functional.softmax(logits, 1)
        np.save(training_args.output_dir+'/probs.npy', probs.numpy())
        log_probs = torch.log(probs)
        np.save(training_args.output_dir+'/log_probs.npy', log_probs.numpy())

    #if training_args.delete_model:
    #    del_model_command = 'find %s -type f -name \'pytorch_model.bin\' -exec rm -rf {} \;'%training_args.output_dir
    #    os.system(del_model_command)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test")
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))

    return results



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
