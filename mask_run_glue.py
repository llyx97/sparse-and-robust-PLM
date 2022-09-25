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
import os, sys, re, json, pickle
from dataclasses import dataclass, field
from typing import Dict, Optional
import utils.param_parser as param_parser

import numpy as np
import torch

import sys
sys.path.append('transformer/src/')

from hg_transformers.configuration_auto import AutoConfig
from hg_transformers.modeling_auto import AutoModelForSequenceClassification
from hg_transformers.tokenization_auto import AutoTokenizer
from hg_transformers.trainer_utils import EvalPrediction
from hg_transformers.data.datasets.glue import GlueDataset
from load_dataset import GlueDataset, MultiDataset
from hg_transformers.hf_argparser import HfArgumentParser
from hg_transformers.training_args import TrainingArguments as BaseTrainingArguments
from hg_transformers.data.processors.glue import glue_output_modes, glue_processors, glue_tasks_num_labels
from hg_transformers.data.metrics import glue_compute_metrics
from hg_transformers.trainer import set_seed
from hg_transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule
#from optimization import AdamW
from hg_transformers.optimization import AdamW
from hg_transformers.mask_trainer import Trainer
import masking.maskers as maskers
import masking.sparsity_control as sp_control
from sklearn.metrics import f1_score
import hg_transformers

from run_glue import load_mask_and_prune


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


def init_masker(conf, model, logger):
    # init the masker scheduler.

    conf.masking_scheduler_conf_ = (
        param_parser.dict_parser(conf.masking_scheduler_conf)
        if conf.masking_scheduler_conf is not None
        else None
    )
    conf.masking_scheduler_conf_['final_sparsity'] = conf.zero_rate
    conf.masking_scheduler_conf_['final_epoch'] = conf.final_sparsity_epoch
    if conf.init_sparsity is not None:
        conf.masking_scheduler_conf_['init_sparsity'] = conf.init_sparsity
    if conf.masking_scheduler_conf is not None:
        for k, v in conf.masking_scheduler_conf_.items():
            setattr(conf, f"masking_scheduler_{k}", v)
    conf.logger = logger

    masker_scheduler = sp_control.MaskerScheduler(conf)

    # init the masker.
    assert not (conf.train_classifier and conf.mask_classifier), "If the classifier is masked, don't train its weights!"
    masker = maskers.Masker(
        masker_scheduler=masker_scheduler,
        logger=logger,
        mask_biases=conf.mask_biases,
        structured_masking_info={
            "structured_masking": conf.structured_masking,
            "structured_masking_types": conf.structured_masking_types,
            "force_masking": conf.force_masking,
        },
        threshold=conf.threshold,
        init_scale=conf.init_scale,
        which_ptl=conf.model_type,
        controlled_init=conf.controlled_init,
        train_classifier=conf.train_classifier,
        global_prune=conf.global_prune,
    )

    # assuming mask all stuff in one transformer block, absorb bert.pooler directly
    #weight_types = ["K", "Q", "V", "AO", "I", "O", "P", "E"]   # Add "E" to mask word embedding
    weight_types = ["K", "Q", "V", "AO", "I", "O", "P"]

    # parse the get the names of layers to be masked.
    assert conf.layers_to_mask is not None, "Please specify which BERT layers to mask."
    conf.layers_to_mask_ = (
        [int(x) for x in conf.layers_to_mask.split(",")]
        if "," in conf.layers_to_mask
        else [int(conf.layers_to_mask)]
    )
    names_tobe_masked = set()
    names_tobe_masked = maskers.chain_module_names(
        conf.model_type, conf.layers_to_mask_, weight_types
    )
    if conf.mask_classifier:
        if conf.model_type == "bert" or conf.model_type == "distilbert":
            names_tobe_masked.add("classifier")
        elif conf.model_type == "roberta":
            if (
                conf.model_scheme == "postagging"
                or conf.model_scheme == "multiplechoice"
            ):
                names_tobe_masked.add("classifier")
            elif conf.model_scheme == "vector_cls_sentence":
                names_tobe_masked.add("classifier.dense")
                names_tobe_masked.add("classifier.out_proj")

    # patch modules.
    masker.patch_modules(
        model=model,
        names_tobe_masked=names_tobe_masked,
        name_of_masker=conf.name_of_masker,
    )
    return masker


@dataclass
class TrainingArguments(BaseTrainingArguments):
    """
    This is a subclass of transformers.TrainingArguments
    """
    best_metric: str = field(
        default='eval_acc', metadata={"help": "The evaluation metric for best checkpoint selection"}
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
    teacher_prob_dir: str = field(
        default=None, metadata={"help": "The directorty of teacher model's predicted probability file."}
    )
    anneal_bias_range: Optional[str] = field(
        default=None, metadata={"help": "The range of bias degree annealing, separated by _."}
    )
    train_subset_size: Optional[int] = field(
        default=0, metadata={"help": "The number of data in the subset for training. If equals to 0, use the entire training set."}
    )
    start_step_ratio: Optional[float] = field(
        default=0.7, metadata={"help": "Control the step from which we start to consider the best result, select from the range [0, 1]."}
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
        default=0., metadata={"help": "The percentate of 0 in model weights."}
    )
    threshold: Optional[float] = field(
        default=1e-2, metadata={"help": "The threshold for masking."}
    )
    init_scale: Optional[float] = field(
        default=2e-2, metadata={"help": "For initialization the real-value mask matrices."}
    )
    mask_classifier: str2bool = field(
        default=False, metadata={"help": "Whether to mask classifier weights."}
    )
    mask_biases: str2bool = field(
        default=False, metadata={"help": "Whether to mask biases."}
    )
    force_masking: Optional[str] = field(
        default='bert', metadata={"help": "?", "choices": ["all", "bert", "classifier"]}
    )
    controlled_init: Optional[str] = field(
        default=None,
        metadata={"help": "To use magnitude pruning or random pruning. mag or rand",
                "choices": ["magnitude", "uniform", "magnitude_and_uniform", "double_uniform", "magnitude_soft"]}
    )
    structured_masking: Optional[str] = field(
        default=None, metadata={"help": "Whether to perform structured masking."}
    )
    structured_masking_types: Optional[str] = field(
        default=None, metadata={"help": "The type of structured masking."}
    )
    name_of_masker: Optional[str] = field(
        default='MaskedLinear1', metadata={"help": "To type of masker to use."}
    )
    layers_to_mask: Optional[str] = field(
        default='0,1,2,3,4,5,6,7,8,9,10,11', metadata={"help": "The layers to mask."}
    )
    masking_scheduler_conf: Optional[str] = field(
        default='lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1',
        metadata={"help": "Configurations for making scheduler."}
    )
    init_sparsity: Optional[float] = field(
        default=None, metadata={"help": "The initial sparsity for sparsity scheduling."}
    )
    final_sparsity_epoch: Optional[float] = field(
        default=1., metadata={"help": "The final epoch for sparsity scheduling."}
    )
    mask_seed: Optional[int] = field(
        default=1, metadata={"help": "The seed for random masking."}
    )
    train_classifier: str2bool = field(
        default=False, metadata={"help": "Whether to train classifier."}
    )
    global_prune: str2bool = field(
        default=False, metadata={"help": "Whether to conduct global pruning"}
    )
    structured: str2bool = field(
        default=False, metadata={"help": "Whether to use structured pruning."}
    )
    train_head_mask: str2bool = field(
        default=False, metadata={"help": "Whether to train head mask."}
    )
    train_ffn_mask: str2bool = field(
        default=False, metadata={"help": "Whether to train FFN mask."}
    )
    load_mask_from: Optional[str] = field(
        default=None, metadata={"help": "The directory to load mask from"}
    )
    model_scheme : Optional[str] = field(
        default="vector_cls_sentence", metadata={"help": "The type of classifier for roberta. Used when mask_classifier is true."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dataset_names: Optional[str] = field(
        default=None, metadata={"help": "The name of datasets, separated by comma."}
    )
    set_types: Optional[str] = field(
        default=None, metadata={"help": "The type of datasets, separated by comma."}
    )
    duplicates: Optional[str] = field(
        default=None, metadata={"help": "The number of times eash dataset is duplicated, separated by comma."}
    )
    synthetic_data: str2bool = field(
        default=False, metadata={"help": "Whether to train and test with synthetic bias data."}
    )
    eval_ood: str2bool = field(
        default=True, metadata={"help": "Whether to evaluate with ood dataset."}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


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

    def load_model():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
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
        return model, config
    model, config = load_model()


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        #model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    # Get datasets
    train_dataset = MultiDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
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

    masker = init_masker(model_args, model, logger)

    for n, p in model.named_parameters():
        print(n, p.requires_grad)

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
    trainer = Trainer(
        model=model,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ood_datasets=ood_datasets,
        compute_metrics=compute_metrics,
        compute_metrics_ood=compute_metrics_ood,
        optimizers=None,
        masker=masker,
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
        # trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        #del_model_command = 'rm -r %s/pytorch_model.bin'%training_args.output_dir
        #os.system(del_model_command)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
        output_eval_file = os.path.join(
                training_args.output_dir, f"best_eval_results_{eval_dataset.args.task_name}.txt"
            )
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
        if training_args.do_train or model_args.load_mask_from is not None:
            logger.info("*** Loading best checkpoint ***")
            model, config = load_model()
            if model_args.load_mask_from is not None:
                    mask_dir = model_args.load_mask_from
            else:
	            mask_dir = os.path.join(training_args.output_dir, 'best_eval_mask')
            model = load_mask_and_prune(mask_dir, model, model_args)
            trainer.model = model.to(training_args.device)
            zero = see_weight_rate(model, model_args.model_type)
            print('model 0:',zero)

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
            logger.info("***** Eval results {} *****".format(task_name))
            for key, value in results.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
                if (key=='eval_mcc' or key=='eval_acc' or key=='eval_pearson') and training_args.do_train:
                    best_score = value if best_score < value else best_score
            try:
                logger.info("  %s = %.4f", 'best_score', best_score)
                writer.write("%s = %.4f\n" % ('best_score', best_score))
            except UnboundLocalError:
                logger.info("This is pure evaluation.")


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
