# -*- coding: utf-8 -*-
import functools

from .maskers import MaskedLinearX


"""target sparsity controller."""


def automated_gradual_sparsity(
    init_sparsity, final_sparsity, interval_epoch, init_epoch, final_epoch
):
    """
    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.
    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    (https://arxiv.org/pdf/1710.01878.pdf)
    """

    def f(current_epoch, current_sparsity):
        if current_epoch > final_epoch:
            return final_sparsity

        # estimate the current sparsity.
        span = final_epoch - init_epoch

        if span != 0:
            target_sparsity = (
                final_sparsity
                + (init_sparsity - final_sparsity)
                * (1.0 - (1.0 * (current_epoch - init_epoch) / span)) ** 3
            )
        else:
            target_sparsity = final_sparsity
        return target_sparsity

    return f


def stepwise_sparsity(
    init_sparsity,
    final_sparsity,
    interval_epoch,
    init_epoch,
    final_epoch,
    sparsity_incremental_ratio,
    with_safety_check=True,
):
    """A stepwise sparsity scheme."""

    def f(current_epoch, current_sparsity):
        if current_epoch < init_epoch:
            return init_sparsity
        elif current_epoch >= final_epoch:
            return final_sparsity
        elif final_epoch > current_epoch >= init_epoch:
            if (current_epoch - init_epoch) % interval_epoch <= 1e-5:
                return (
                    current_sparsity
                    + (1 - current_sparsity) * sparsity_incremental_ratio
                )
            else:
                return current_sparsity

    def get_actual_final_sparsity():
        current_sparsity = init_sparsity
        for current_epoch in range(init_epoch, final_epoch, interval_epoch):
            current_sparsity = f(current_epoch, current_sparsity)
        if (final_epoch - init_epoch) % interval_epoch <= 1e-5:
            current_sparsity += (1 - current_sparsity) * sparsity_incremental_ratio
        return current_sparsity

    if with_safety_check:
        actual_final_sparsity = get_actual_final_sparsity()
        if actual_final_sparsity < final_sparsity:
            raise ValueError(
                "Increase initial sparsity and/or incremental ratio,"
                + "current final sparsity is {}, required value is {}".format(
                    actual_final_sparsity, final_sparsity
                )
            )

    return f


class MaskerScheduler(object):
    def __init__(self, conf):
        # init.
        self.conf = conf
        self.masking_scheduler_conf_ = conf.masking_scheduler_conf_
        self._current_sparsity = 0

        if conf.masking_scheduler_conf_ is not None:
            assert "final_sparsity" in self.masking_scheduler_conf_
            assert "sparsity_warmup_interval_epoch" in self.masking_scheduler_conf_
            self.init_sparsity = (
                self.masking_scheduler_conf_["init_sparsity"]
                if "init_sparsity" in self.masking_scheduler_conf_
                else self.masking_scheduler_conf_["final_sparsity"]
            )
            self.get_sparsity_fn = self._get_pruner()
        else:
            self.init_sparsity = 0.5
            self.get_sparsity_fn = None

    @property
    def is_skip(self):
        if self.get_sparsity_fn is None or (
            "lambdas_lr" in self.conf.masking_scheduler_conf_
            and self.conf.masking_scheduler_conf_["lambdas_lr"] == 0
        ):
            return True
        else:
            return False

    def _get_pruner(self):
        if (
            "sparsity_warmup" not in self.masking_scheduler_conf_
            or self.masking_scheduler_conf_["sparsity_warmup"]
            == "automated_gradual_sparsity"
        ):
            self.conf.logger.info("use automated_gradual_sparsity.")
            return automated_gradual_sparsity(
                init_sparsity=self.init_sparsity,
                final_sparsity=self.masking_scheduler_conf_["final_sparsity"],
                interval_epoch=self.masking_scheduler_conf_[
                    "sparsity_warmup_interval_epoch"
                ],
                init_epoch=int(self.conf.num_epochs * 0.1)
                if "init_epoch" not in self.masking_scheduler_conf_
                else self.masking_scheduler_conf_["init_epoch"],
                final_epoch=int(self.conf.num_epochs * 0.8)
                if "final_epoch" not in self.masking_scheduler_conf_
                else self.masking_scheduler_conf_["final_epoch"],
            )
        elif self.masking_scheduler_conf_["sparsity_warmup"] == "stepwise_sparsity":
            self.conf.logger.info("use stepwise pruner.")
            assert "sparsity_incremental_ratio" in self.masking_scheduler_conf_
            return stepwise_sparsity(
                init_sparsity=self.init_sparsity,
                final_sparsity=self.masking_scheduler_conf_["final_sparsity"],
                interval_epoch=self.masking_scheduler_conf_[
                    "sparsity_warmup_interval_epoch"
                ],
                init_epoch=int(self.conf.num_epochs * 0.1)
                if "init_epoch" not in self.masking_scheduler_conf_
                else self.masking_scheduler_conf_["init_epoch"],
                final_epoch=int(self.conf.num_epochs * 0.8)
                if "final_epoch" not in self.masking_scheduler_conf_
                else self.masking_scheduler_conf_["final_epoch"],
                sparsity_incremental_ratio=self.masking_scheduler_conf_[
                    "sparsity_incremental_ratio"
                ],
            )
        else:
            raise NotImplementedError

    def step(self, cur_epoch):
        self.cur_epoch = cur_epoch

        # get target_sparsity under the init and final sparsity constraint.
        _target_sparsity = self.get_sparsity_fn(cur_epoch, self._current_sparsity)

        if self.masking_scheduler_conf_["final_sparsity"] > self.init_sparsity:
            min_sparsity = self.init_sparsity
            max_sparsity = self.masking_scheduler_conf_["final_sparsity"]
        else:
            max_sparsity = self.init_sparsity
            min_sparsity = self.masking_scheduler_conf_["final_sparsity"]
        self.target_sparsity = min(max_sparsity, max(_target_sparsity, min_sparsity))

        # get the _incremental_sparsity_ratio based on the current sparsity.
        _incremental_sparsity = (self.target_sparsity - self._current_sparsity) / (
            1 - self._current_sparsity
        )
        return _incremental_sparsity, self.target_sparsity, self.is_sparsity_change()

    def is_meet_sparsity(self):
        if self.target_sparsity >= self.masking_scheduler_conf_["final_sparsity"]:
            return True
        else:
            return False

    def is_sparsity_change(self):
        if self._current_sparsity == self.target_sparsity:
            return False
        else:
            self._current_sparsity = self.target_sparsity
            return True

    def get_sparsity_over_whole_model(self, model, masker, trainable=True):
        def get_modified_linear_modules(my_module):
            modules = []
            for m in my_module.children():
                if isinstance(m, MaskedLinearX):
                    modules.append(m)
                else:
                    modules.extend(get_modified_linear_modules(m))
            return modules

        def get_info_from_one_layer(masks, tensor_name, info_type):
            mask = masks[0 if tensor_name == "weight" else 1]
            if mask is not None:
                if info_type == "nnz":
                    return mask.sum()
                elif info_type == "tot":
                    return mask.numel()
                else:
                    raise NotImplementedError(
                        f"the info_type={info_type} is not supported yet."
                    )
            else:
                return None

        # get modified linear modules
        modified_linear_modules = get_modified_linear_modules(model)

        # get the masks as well as the corresponding information in these modified linear modules.
        masks = [module.get_masks() for module in modified_linear_modules]
        nnz_info = [
            get_info_from_one_layer(mask, tensor_name="weight", info_type="nnz")
            for mask in masks
        ] + [
            get_info_from_one_layer(mask, tensor_name="bias", info_type="nnz")
            for mask in masks
        ]
        tot_info = [
            get_info_from_one_layer(mask, tensor_name="weight", info_type="tot")
            for mask in masks
        ] + [
            get_info_from_one_layer(mask, tensor_name="bias", info_type="tot")
            for mask in masks
        ]

        # evaluate the overall sparsity.
        total_nnz = sum([x for x in nnz_info if x is not None])
        total_tot = sum([x for x in tot_info if x is not None])
        return 1 - total_nnz / total_tot
