# -*- coding: utf-8 -*-
import math
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""get the names for ptl layers."""
_bert_roberta_names = {
    "K": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.key",
    "Q": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.query",
    "V": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.value",
    "AO": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.output.dense",
    "I": lambda ptl, l: f"{ptl}.encoder.layer.{l}.intermediate.dense",
    "O": lambda ptl, l: f"{ptl}.encoder.layer.{l}.output.dense",
    "P": lambda ptl, l: f"{ptl}.pooler.dense",
}

_distilbert_names = {
    "K": lambda _, l: f"distilbert.transformer.layer.{l}.attention.k_lin",
    "Q": lambda _, l: f"distilbert.transformer.layer.{l}.attention.q_lin",
    "V": lambda _, l: f"distilbert.transformer.layer.{l}.attention.v_lin",
    "AO": lambda _, l: f"distilbert.transformer.layer.{l}.attention.out_lin",
    "I": lambda _, l: f"distilbert.transformer.layer.{l}.ffn.lin1",
    "O": lambda _, l: f"distilbert.transformer.layer.{l}.ffn.lin2",
    "P": lambda _, l: f"pre_classifier",
}


def chain_module_names(which_ptl, layer_idices, abbres):
    if which_ptl == "distilbert":
        _plt_names = _distilbert_names
    else:
        _plt_names = _bert_roberta_names
    names = set()
    for abbre in abbres:
        for l in layer_idices:
            names.add(_plt_names[abbre](which_ptl, l))
    return names


class MaskedLinearX(nn.Module):
    def __init__(
        self, scheme_idx, weight, bias, mask_biases, **kwargs,
    ):
        super(MaskedLinearX, self).__init__()
        init_scale = kwargs["init_scale"] if "init_scale" in kwargs else None
        init_sparsity = kwargs["init_sparsity"] if "init_sparsity" in kwargs else None
        self.name = kwargs["name"] if "name" in kwargs else None
        self.threshold = kwargs["threshold"] if "threshold" in kwargs else None
        self.threshold_fn = _scheme_idx_to_fn[scheme_idx]().apply
        self.mask_biases = mask_biases

        self.weight = weight
        self.bias = bias
        self.structured_mask_expanding = None
        self._controlled_init = (
            kwargs["controlled_init"] if "controlled_init" in kwargs else False
        )

        # init the masks (with the options of structured_pruning).
        assert "structured_masking_info" in kwargs
        structured_masking_info = kwargs["structured_masking_info"]
        structured_masking = structured_masking_info["structured_masking"]
        structured_masking_types = structured_masking_info["structured_masking_types"]
        self.force_masking = structured_masking_info["force_masking"]
        self.is_structured_masking = (
            structured_masking is not None and structured_masking != "None"
        )
        self.meet_structured_masking_cond = structured_masking_types is None or any(
            [
                True if _type in self.name else False
                for _type in structured_masking_types
            ]
        )

        # adjust the random initialization scale of the mask to satisify the initial sparsity.
        init_scales = self.get_init_scales(scheme_idx, init_sparsity, init_scale)

        # init the masks for structured pruning.
        self.structured_masked = False
        if self.is_structured_masking:
            # either (1) apply structured pruning to all layers,
            # or (2) apply structured pruning to some specific layers (e.g. `self`) and apply unstructured pruning to left layers,
            # or (3) only apply structured pruning to some specific layers (e.g., `self`).
            if self.meet_structured_masking_cond:
                self.structured_masked = True
                if structured_masking == "layers":
                    _template = torch.FloatTensor([1]).uniform_(*init_scales)
                    self.weight_mask = nn.Parameter(_template.clone())
                    if mask_biases:
                        self.bias_mask = nn.Parameter(_template.clone())
                elif structured_masking == "heads":
                    # Linear.weight -- the learnable weights of the module of shape (out_features, in_features).
                    # num_attention_heads = config.num_attention_heads
                    # attention_head_size = int(config.hidden_size / config.num_attention_heads)
                    # all_head_size = num_attention_heads * attention_head_size
                    #   then e.g., query = nn.Linear(config.hidden_size, all_head_size)
                    #   but query.weight = (all_head_size, config.hidden_size)
                    assert "self" in self.name

                    # get ptl model info.
                    conf = structured_masking_info["ptl_config"]
                    num_attention_heads = conf.num_attention_heads
                    attention_head_size = int(conf.hidden_size / num_attention_heads)

                    # init the masks.
                    _template = torch.FloatTensor([1] * num_attention_heads).uniform_(
                        *init_scales
                    )
                    self.structured_mask_expanding = nn.Parameter(
                        torch.ones(num_attention_heads, attention_head_size),
                        requires_grad=False,
                    )
                    self.weight_mask = nn.Parameter(_template.clone())
                    if mask_biases:
                        self.bias_mask = nn.Parameter(_template.clone())
                else:
                    raise NotImplementedError(
                        f"structured_masking={structured_masking} not supported yet"
                    )

        # init the masks for unstructured pruning.
        self.unstructured_masked = False
        if not self.structured_masked and (
            not self.is_structured_masking or self.force_masking in self.name
        ):
            self.unstructured_masked = True

            # either randomly and uniformly initialize the mask.
            if self._controlled_init is None:
                self.weight_mask = nn.Parameter(
                    torch.empty_like(self.weight).uniform_(*init_scales)
                )
                if mask_biases:
                    self.bias_mask = nn.Parameter(
                        torch.empty_like(self.bias).uniform_(*init_scales)
                    )
            # or by the corresponding magnitude.
            else:
                self.weight_mask = self.controlled_init(
                    self.weight,
                    init_sparsity,
                    self.threshold,
                    controlled_init_type=self._controlled_init,
                )
                if mask_biases:
                    self.bias_mask = self.controlled_init(
                        self.bias,
                        init_sparsity,
                        self.threshold,
                        controlled_init_type=self._controlled_init,
                    )

    def controlled_init(self, weight, init_sparsity, threshold, controlled_init_type):
        # get the threshold by magnitude.
        _weight_size = weight.nelement()
        _num_zero_element = int(_weight_size * init_sparsity)

        def _magnitude():
            _weight = torch.zeros_like(weight)
            _weight_abs = weight.abs()
            _flatten_weight = _weight_abs.view(-1)

            _bool_masks = (
                _weight_abs
                > torch.kthvalue(input=_flatten_weight, k=_num_zero_element).values
            )
            _weight[_bool_masks] = 2.0 * threshold
            _weight[~_bool_masks] = 0.0 * threshold
            return _weight

        def _uniform():
            _weight = torch.zeros_like(weight.view(-1))
            indices = np.arange(_weight_size)
            sampled_indices = np.random.choice(indices, size=_num_zero_element)
            _bool_masks = torch.ones_like(_weight)
            _bool_masks[sampled_indices] = 0
            _bool_masks = _bool_masks.bool()

            _weight[_bool_masks] = 2.0 * threshold
            _weight[~_bool_masks] = 0.0 * threshold
            return _weight.view(*weight.size())

        def _double_uniform():
            #
            _weight = torch.zeros_like(weight.view(-1))
            indices = np.arange(_weight_size)
            sampled_indices = np.random.choice(indices, size=_num_zero_element)
            _bool_masks = torch.ones_like(_weight)
            _bool_masks[sampled_indices] = 0
            _bool_masks = _bool_masks.bool()

            #
            _above_weight = _weight.clone()
            _below_weight = _weight.clone()
            _above_weight.uniform_(1.1 * threshold, 1.5 * threshold).mul_(_bool_masks)
            _below_weight.uniform_(0.5 * threshold, 0.9 * threshold).mul_(~_bool_masks)
            _weight = _above_weight + _below_weight
            return _weight.view(*weight.size())

        if controlled_init_type == "magnitude":
            # it will sample indices from the tensor (based on the magnitude)
            # to assign values for the mask.
            weight_mask = _magnitude()
        elif controlled_init_type == "uniform":
            # it will randomly sample indices from the tensor
            # to assign values for the mask.
            weight_mask = _uniform()
        elif controlled_init_type == "magnitude_and_uniform":
            # for linear layers in bert, we use _magnitude
            # for final linear layer, we use uniform.
            if "bert" in self.name:
                weight_mask = _magnitude()
            else:
                weight_mask = _uniform()
        elif controlled_init_type == "double_uniform":
            # it will randomly (and uniformly) sample indices from the tensor
            # to assign uniformly sampled values for the mask.
            weight_mask = _double_uniform()
        else:
            raise NotImplementedError("this controlled init type is not supported.")
        return nn.Parameter(weight_mask)

    def get_init_scales(self, scheme_idx, init_sparsity, init_scale):
        if scheme_idx == "MaskedLinear1":
            s = (init_scale + self.threshold) / init_sparsity - init_scale
            return (-init_scale, s)
        elif scheme_idx == "MaskedLinear2":
            warnings.warn(f"we cannot control the initial sparsity for {scheme_idx}.")
            return (-init_scale, init_scale)
        elif scheme_idx == "MaskedLinear3":
            p = 1 - init_sparsity
            i_s = math.log(p / (1 - p))
            return (i_s, i_s)
        else:
            return (-init_scale, init_scale)

    def forward(self, x):
        raise NotImplementedError


"""functions to mask the bert."""

# structured_pruning assistant.


def reshape_mask_for_sp(mask, structured_mask_expanding, name="weight"):
    # a hack for structured pruning.
    if structured_mask_expanding is not None:
        # nn.Linear(config.hidden_size, num_attention_heads * attention_head_size)
        # weight_mask: [num_attention_heads]
        # structured_mask_expanding = [num_attention_heads, attention_head_size]
        # weight_mask * structured_mask_expanding -> [num_attention_heads, attention_head_size]
        # reshape mask to [num_attention_heads * attention_head_size, 1]
        _mask = (mask.unsqueeze(1) * structured_mask_expanding).view(-1)
        if name == "weight":
            mask = _mask.unsqueeze(1)
        elif name == "bias":
            mask = _mask.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError("not supported mask type.")
    return mask


# masking scheme 0, i.e., no masks:
class MaskedLinear0(nn.Module):
    def __init__(self, weight, bias, **kwargs):
        super(MaskedLinear0, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# masking scheme 1:
# use $ \frac{\partial \cL}{\partial \mm} $ to approximate $ \frac{\partial \cL}{\partial \mm^r} $


def binarizer_fn1(inputs, threshold):
    outputs = inputs.clone()
    outputs[inputs.le(threshold)] = 0.0
    outputs[inputs.gt(threshold)] = 1.0
    return outputs


class _Binarizer1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        return binarizer_fn1(inputs, threshold)

    @staticmethod
    def backward(ctx, gradOutput):
        return (gradOutput, None, None)


class MaskedLinear1(MaskedLinearX):
    def __init__(self, weight, bias, mask_biases, **kwargs):
        super(MaskedLinear1, self).__init__(
            "MaskedLinear1", weight, bias, mask_biases, **kwargs
        )

    def get_masks(self):
        M_w = self.threshold_fn(self.weight_mask, self.threshold)
        M_w = reshape_mask_for_sp(M_w, self.structured_mask_expanding, name="weight")

        if self.mask_biases:
            M_b = self.threshold_fn(self.bias_mask, self.threshold)
            M_b = reshape_mask_for_sp(M_b, self.structured_mask_expanding, name="bias")
        else:
            M_b = None
        return M_w, M_b

    def forward(self, x):
        M_w, M_b = self.get_masks()

        if M_b is not None:
            return F.linear(x, self.weight * M_w, self.bias * M_b)
        return F.linear(x, self.weight * M_w, self.bias)


# masking scheme 2:
# evaluate the explicit $ \frac{\partial \cL}{\partial \mm^r} $.


def binarizer_fn2(inputs):
    outputs = inputs.clone()
    inputs.data.clamp_(-1, 1)
    outputs.data = (torch.sign(outputs.data) + 1) / 2
    return outputs


class _Binarizer2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return binarizer_fn2(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        inputs, *_ = ctx.saved_variables
        gradOutput[inputs.ge(1)] = 0
        gradOutput[inputs.le(-1)] = 0
        return gradOutput


class MaskedLinear2(MaskedLinearX):
    def __init__(self, weight, bias, mask_biases, **kwargs):
        super(MaskedLinear2, self).__init__(
            "MaskedLinear2", weight, bias, mask_biases, **kwargs
        )

    def get_masks(self):
        M_w = self.threshold_fn(self.weight_mask)
        M_w = reshape_mask_for_sp(M_w, self.structured_mask_expanding, name="weight")

        if self.mask_biases:
            M_b = self.threshold_fn(self.bias_mask)
            M_b = reshape_mask_for_sp(M_b, self.sp_expanding, name="bias")
        else:
            M_b = None
        return M_w, M_b

    def forward(self, x):
        M_w, M_b = self.get_masks()

        if M_b is not None:
            return F.linear(x, self.weight * M_w, self.bias * M_b)
        return F.linear(x, self.weight * M_w, self.bias)


# masking scheme 3:
# use the bernoulli sampler


def binarizer_fn3(inputs):
    # outputs = inputs.clone()
    # outputs = torch.bernoulli(torch.sigmoid(outputs))
    outputs = torch.bernoulli(torch.sigmoid(inputs))
    return outputs


class _Binarizer3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return binarizer_fn3(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput


class MaskedLinear3(MaskedLinearX):
    def __init__(self, weight, bias, mask_biases, **kwargs):
        super(MaskedLinear3, self).__init__(
            "MaskedLinear3", weight, bias, mask_biases, **kwargs
        )

    def get_masks(self):
        M_w = self.threshold_fn(self.weight_mask)
        M_w = reshape_mask_for_sp(M_w, self.structured_mask_expanding, name="weight")

        if self.mask_biases:
            M_b = self.threshold_fn(self.bias_mask)
            M_b = reshape_mask_for_sp(M_b, self.sp_expanding, name="bias")
        else:
            M_b = None
        return M_w, M_b

    def forward(self, x):
        M_w, M_b = self.get_masks()

        if M_b is not None:
            return F.linear(x, self.weight * M_w, self.bias * M_b)
        return F.linear(x, self.weight * M_w, self.bias)


_scheme_idx_to_fn = {
    "MaskedLinear1": _Binarizer1,
    "MaskedLinear2": _Binarizer2,
    "MaskedLinear3": _Binarizer3,
}

# a general masker that can use different masking schemes.


def _get_nnz_from(tensor):
    return tensor.char().sum().detach().cpu().numpy()


class Masker(object):
    def __init__(
        self,
        masker_scheduler,
        logger,
        mask_biases,
        structured_masking_info,
        threshold,
        init_scale,
        which_ptl,
        controlled_init,
    ):
        self.masker_scheduler = masker_scheduler
        self.mask_biases = mask_biases
        self.structured_masking_info = structured_masking_info
        self.logger = logger
        self.which_ptl = which_ptl

        self.threshold = torch.tensor(threshold)
        self.init_scale = init_scale
        self.controlled_init = controlled_init

        # init for some comparisons.
        self.init_masks = {}

    def patch_modules(self, model, names_tobe_masked, name_of_masker="MaskedLinear1"):
        self.ptl_config = eval(f"model.{self.which_ptl}.config")
        masked_linear_cls = eval(name_of_masker)
        self.replace(
            model,
            root_name="",
            names_tobe_masked=names_tobe_masked,
            masked_linear_cls=masked_linear_cls,
        )
        self.masked_linear_cls = masked_linear_cls

        # check status and logging.
        # self.log_fn("Check the trainable status.")
        # for _name, param in model.named_parameters():
        #     if param.requires_grad is True:
        #         self.log_fn(f"\t {_name} is trainable.")
        #     else:
        #         self.log_fn(f"\t {_name} is not trainable.")

        self.logger.info("Check the masking status.")
        for m_name, m in model.named_modules():
            if m_name not in names_tobe_masked:
                continue
            if isinstance(m, masked_linear_cls):
                # sparsity check.
                param_info = {}
                for _name, param in m.named_parameters():
                    if "mask" in _name:
                        param_info[_name] = 1.0 - (
                            _get_nnz_from(
                                self.eval_binarizer_fn(
                                    name_of_masker, param, self.threshold
                                )
                            )
                            / np.prod(param.shape)
                        )
                        self.init_masks[f"{m_name}_{_name}"] = self.eval_binarizer_fn(
                            name_of_masker, param.clone().data.cpu(), self.threshold
                        )
                self.logger.info(f"\t {m_name} is MASKED -> {json.dumps(param_info)}")
            else:
                self.logger.info(f"\t {m_name} is NOT MASKED")

    @staticmethod
    def eval_binarizer_fn(name_of_masker, param, threshold):
        if "MaskedLinear1" == name_of_masker:
            return binarizer_fn1(param, threshold)
        elif "MaskedLinear2" == name_of_masker:
            return binarizer_fn2(param)
        elif "MaskedLinear3" == name_of_masker:
            return binarizer_fn3(param)
        else:
            raise NotImplementedError(f"incorrect name_of_masker={name_of_masker}.")

    def replace(self, m, root_name, names_tobe_masked, masked_linear_cls):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if isinstance(target_attr, nn.Module):
                name = root_name + "." + attr_str if len(root_name) > 0 else attr_str
                if hasattr(target_attr, 'weight') and not 'predictions' in name:
                    target_attr.weight.requires_grad = False
                if hasattr(target_attr, 'bias') and not 'predictions' in name:
                    if target_attr.bias is not None:
                        target_attr.bias.requires_grad = False
            if type(target_attr) == nn.Linear:
                masked = False

                if name in names_tobe_masked:
                    # turn off the trainable weights and biases.
                    masked_linear = masked_linear_cls(
                        name=name,
                        weight=target_attr.weight,
                        bias=target_attr.bias,
                        mask_biases=self.mask_biases,
                        threshold=self.threshold,
                        init_sparsity=self.masker_scheduler.init_sparsity,
                        init_scale=self.init_scale,
                        controlled_init=self.controlled_init,
                        structured_masking_info={
                            "ptl_config": self.ptl_config,
                            **self.structured_masking_info,
                        },
                    )
                    # disable the trainability of the original weights in the replaced linear layer.
                    for _name, param in masked_linear.named_parameters():
                        if "mask" not in _name:
                            param.requires_grad = False

                    # displaying the pruning status.
                    if (
                        masked_linear.unstructured_masked
                        or masked_linear.structured_masked
                    ):
                        masked = True
                        # override original linear layer.
                        setattr(m, attr_str, masked_linear)
                        self.logger.info(
                            f"""\t {name} is MASKED: {f"structured masking for layer type={self.structured_masking_info['structured_masking_types']}" if masked_linear.structured_masked else "unstructured masking"}"""
                        )
                if not masked:
                    self.logger.info(f"\t {name} is NOT MASKED")

        for sub_modules_name, sub_modules in m.named_children():
            self.replace(
                m=sub_modules,
                root_name=root_name + "." + sub_modules_name
                if len(root_name) > 0
                else sub_modules_name,
                names_tobe_masked=names_tobe_masked,
                masked_linear_cls=masked_linear_cls,
            )
