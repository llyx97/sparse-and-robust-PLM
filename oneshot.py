import argparse
from hg_transformers import BertConfig, RobertaConfig, set_seed
import torch.nn.utils.prune as prune
import numpy as np  
import torch, os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--weight', default='pre', type=str, help='file_dir')
parser.add_argument('--model', default='glue', type=str, help='file_dir')
parser.add_argument('--rate', default=0.7, type=float, help='rate')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--prune_global', default=True, type=str2bool, help='global pruning or local pruning')
parser.add_argument('--prun_type', default='magnitude', type=str, help='pruning type')
parser.add_argument('--output_dir', default='', type=str, help='output dir')
parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str, help='file_dir')
parser.add_argument('--model_type', default='bert', type=str, help='model type')
args = parser.parse_args()


if args.model_type=='bert':
    from hg_transformers import BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering
elif args.model_type=='roberta':
    from hg_transformers import RobertaForMaskedLM as BertForMaskedLM
    from hg_transformers import RobertaForSequenceClassification as BertForSequenceClassification
    from hg_transformers import RobertaForQuestionAnswering as BertForQuestionAnswering


def pruning_model_local(model,px):
    print('Start %s pruning with zero rate %.2f'%(args.prun_type, args.rate))
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
            if args.prun_type=='magnitude':
                prune.l1_unstructured(module, 'weight', amount=px)
            elif args.prun_type=='random':
                prune.random_unstructured(module, 'weight', amount=px)
    #if args.prun_type=='magnitude':
    #    prune.l1_unstructured(model.embeddings.word_embeddings, 'weight', amount=px)
    #elif args.prun_type=='random':
    #    prune.random_unstructured(model.embeddings.word_embeddings, 'weight', amount=px)


def pruning_model_global(model,px):
    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
    #parameters_to_prune.append((model.bert.embeddings.word_embeddings, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    if args.prun_type=='magnitude':
        prune_method = prune.L1Unstructured
    elif args.prun_type=='random':
        prune_method = prune.RandomUnstructured
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune_method,
        amount=px,
    )


def see_weight_rate(model, model_type):
    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask'] == 0))

    sum_list = sum_list+float(model.state_dict()['%s.pooler.dense.weight_mask'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.pooler.dense.weight_mask'%model_type] == 0))
    #sum_list = sum_list+float(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type].nelement())
    #zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type] == 0))

    return 100*zero_sum/sum_list


set_seed(args.seed)

if args.model_type=='bert':
    config = BertConfig.from_pretrained(
            args.model_name_or_path
    )
elif args.model_type=='roberta':
    config = RobertaConfig.from_pretrained(
            args.model_name_or_path
    )


if args.model == 'glue':

    if args.weight == 'rand':
        print('random')
        model = BertForSequenceClassification(config=config)
        output = 'random_prun/'

    elif args.weight == 'pre':
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
        output = 'pretrain_prun/local'

    local_or_global = 'global' if args.prune_global else 'local'
    output_mask_dir = os.path.join(args.output_dir, args.prun_type, local_or_global, str(args.rate))
    if args.prun_type == 'random':
        output_mask_dir = os.path.join(output_mask_dir, str(args.seed))
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    if args.model_type=='bert':
        bert_model = model.bert
    elif args.model_type=='roberta':
        bert_model = model.roberta

    if args.prune_global:
        pruning_model_global(model, args.rate)
    else:
        pruning_model_local(bert_model, args.rate)
    zero = see_weight_rate(model, args.model_type)
    print('zero rate', zero)

    mask_dict = {}
    weight_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask' in key:
            mask_dict[key] = model_dict[key].bool()
        else:
            weight_dict[key] = model_dict[key]

    print('Saving mask to %s'%output_mask_dir)
    torch.save(mask_dict, os.path.join(output_mask_dir, 'mask.pt'))
    #torch.save(weight_dict, output+'weight.pt')

elif args.model == 'squad':

    if args.weight == 'rand':
        print('random')
        model = BertForQuestionAnswering(config=config)
        output = 'random_prun/'

    elif args.weight == 'pre':
        model = BertForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
        output = 'pretrain_prun/'

    if args.model_type=='bert':
        bert_model = model.bert
    elif args.model_type=='roberta':
        bert_model = model.roberta
    pruning_model(bert_model, args.rate)
    zero = see_weight_rate(model, args.model_type)
    print('zero rate', zero)

    mask_dict = {}
    weight_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask' in key:
            mask_dict[key] = model_dict[key]
        else:
            weight_dict[key] = model_dict[key]

    torch.save(mask_dict, output+'mask.pt')
    torch.save(weight_dict, output+'weight.pt')

elif args.model == 'pretrain':

    if args.weight == 'rand':
        print('random')
        model = BertForMaskedLM(config=config)
        output = 'random_prun/'

    elif args.weight == 'pre':
        model = BertForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
        output = 'pretrain_prun/'

    local_or_global = 'global' if args.prune_global else 'local'
    output_mask_dir = os.path.join(args.output_dir, args.prun_type, local_or_global, str(args.rate))
    if args.prun_type == 'random':
        output_mask_dir = os.path.join(output_mask_dir, str(args.seed))
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    if args.model_type=='bert':
        bert_model = model.bert
    elif args.model_type=='roberta':
        bert_model = model.roberta

    if args.prune_global:
        pruning_model_global(model, args.rate)
    else:
        pruning_model_local(bert_model, args.rate)

    zero = see_weight_rate(model, args.model_type)
    print('zero rate', zero)

    mask_dict = {}
    weight_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask' in key:
            mask_dict[key] = model_dict[key]
        else:
            weight_dict[key] = model_dict[key]

    print('Saving mask to %s'%output_mask_dir)
    torch.save(mask_dict, os.path.join(output_mask_dir, 'mask.pt'))
    #torch.save(weight_dict, os.path.join(output, 'weight.pt'))
