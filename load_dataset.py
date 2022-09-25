import numpy as np
import logging, json, time, torch, jsonlines
from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from filelock import FileLock
from os.path import join, exists
from torch.utils.data.dataset import Dataset
from hg_transformers.data.processors.glue import glue_convert_examples_to_features, glue_output_modes, QqpProcessor, MnliProcessor, MnliMismatchedProcessor
from hg_transformers.data.processors.glue import FeverProcessor as FeverProcessorGlue
from hg_transformers.data.datasets.glue import GlueDataTrainingArguments
from hg_transformers.data.processors.utils import InputExample, DataProcessor
from hg_transformers.data.data_collator import DataCollator, InputDataClass
from hg_transformers import PreTrainedTokenizer
from hg_transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from hg_transformers.tokenization_xlm_roberta import XLMRobertaTokenizer


logger = logging.getLogger(__name__)

@dataclass
class DataCollatorWithBias(DataCollator):
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    def collate_batch(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        # In this method we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if hasattr(first, "label") and first.label is not None:
            if type(first.label) is int:
                labels = torch.tensor([f.label for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label for f in features], dtype=torch.float)
            batch = {"labels": labels}
        elif hasattr(first, "label_ids") and first.label_ids is not None:
            if type(first.label_ids[0]) is int:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
            batch = {"labels": labels}
        else:
            batch = {}

        # Handling of all other possible attributes.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in vars(first).items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if (k=='bias' and first.bias is not None) or (k=='teacher_probs' and first.teacher_probs is not None):
                    batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.float)
                else:
                    batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        return batch

@dataclass(frozen=False)
class InputFeatures:
    """
    An extension of the transformers.InputFeatures w/ bias degree of each example.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        bias: (Optional) Bias degree  corresponding to the input.
    """
    input_ids: List[int]
    example_id: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    bias: Optional[Union[int, float]] = None
    teacher_probs: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

class FeverProcessor(DataProcessor):

    def _read_jsonlines(self, input_file):
        lines = []
        with open(input_file, "r", encoding='utf-8') as f:
            reader = jsonlines.Reader(f)
            for line in reader.iter(type=dict):
                lines.append(line)

        return lines

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(join(data_dir, "dev.jsonl")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(join(data_dir, "test.jsonl")))

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        target_labels = self.get_labels()
        num_labels = len(target_labels)
        for (i, line) in enumerate(lines):
            guid = line['id']
            text_a = line['claim']
            if 'evidence' in line:
                text_b = line['evidence']
            else:
                text_b = line['evidence_sentence']
            if 'gold_label' in line:
                label = line['gold_label']
            else:
                label = line['label']
            if 'weight' in line:
                weight = line['weight']
            else:
                weight = 0.0

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                #InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, weight=weight))
        return examples


class PawsProcessor(DataProcessor):
    """Processor for the PAWS data set (GLUE version)."""

    def get_test_examples(self, data_dir, file_name='test.tsv'):
        """See base class."""
        return self._create_examples(data_dir, file_name)

    def get_dev_examples(self, data_dir, file_name='dev.tsv'):
        """See base class."""
        return self._create_examples(data_dir, file_name)

    def get_train_examples(self, data_dir, file_name='train.tsv'):
        """See base class."""
        return self._create_examples(data_dir, file_name)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, data_path, file_name):
        """Creates examples for the training, dev and test sets."""
        examples = []
        logging.info("Loading paws...")
        src = join(data_path, file_name)
        #if not exists(src):
        #    logging.info("Downloading source to %s..." % data_path)
        #    py_utils.download_to_file(HANS_URL, src)

        with open(src, "r") as f:
            f.readline()
            lines = f.readlines()
        lines = self._read_tsv(src)

        q1_index = 1
        q2_index = 2

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            pair_id = line[0]
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[3]
            except IndexError:
                continue
            examples.append(InputExample(guid=pair_id, text_a=text_a, text_b=text_b, label=label))
        return examples


class HansProcessor():
    """Processor for the HANS data set (GLUE version)."""

    def get_test_examples(self, data_dir, file_name="heuristics_evaluation_set.txt"):
        """See base class."""
        return self._create_examples(data_dir, file_name)

    def get_train_examples(self, data_dir, file_name="heuristics_train_set.txt"):
        """See base class."""
        return self._create_examples(data_dir, file_name)

    def get_labels(self):
        """See base class."""
        return ["non-entailment", "entailment"]

    def _create_examples(self, data_path, file_name, n_samples=None):
        """Creates examples for the training, dev and test sets."""
        examples = []
        logging.info("Loading hans...")
        src = join(data_path, file_name)
        if not exists(src):
            logging.info("Downloading source to %s..." % data_path)
            py_utils.download_to_file(HANS_URL, src)

        with open(src, "r") as f:
            f.readline()
            lines = f.readlines()

        if n_samples is not None:
            lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples, replace=False)

        for line in lines:
            parts = line.split("\t")
            label = parts[0]
            s1, s2, pair_id = parts[5:8]
            examples.append(InputExample(guid=pair_id, text_a=s1, text_b=s2, label=label))
        return examples


class MnliSynProcessor(DataProcessor):
    """Processor for the MultiNLI data set with synthetic bias (GLUE version)."""

    def __init__(self, args):
        self.args = args
        self.n_classes = len(self.get_labels())

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        if set_type=='train':
            syn_example_ids = np.random.choice([i for i in range(1, len(lines))], size=int(self.args.synthetic_rate*len(lines)), replace=False)
            biased_example_ids = np.random.choice(syn_example_ids, size=int(self.args.bias_rate*len(syn_example_ids)), replace=False)
            anti_biased_example_ids = list(set(syn_example_ids) - set(biased_example_ids))
        else:
            # Eliminate the biased feature in the dev/test sets
            syn_example_ids = [i for i in range(1, len(lines))]
            biased_example_ids = []
            anti_biased_example_ids = syn_example_ids
        label_map = {label: i for i, label in enumerate(self.get_labels())}

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            if i in syn_example_ids and not set_type.startswith("test"):
                if i in biased_example_ids:
                    text_b = str(label_map[label]) + ' ' + text_b
                elif i in anti_biased_example_ids:
                    noise = (int(label_map[label]) + np.random.randint(1, self.n_classes)) % self.n_classes  # Select a different class
                    text_b = str(noise) + ' ' + text_b
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliMismatchedSynProcessor(MnliSynProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class QqpSynProcessor(DataProcessor):
    """Processor for the QQP data set with synthetic bias (GLUE version)."""

    def __init__(self, args):
        self.args = args
        self.n_classes = len(self.get_labels())

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []

        if set_type=='train':
            syn_example_ids = np.random.choice([i for i in range(1, len(lines))], size=int(self.args.synthetic_rate*len(lines)), replace=False)
            biased_example_ids = np.random.choice(syn_example_ids, size=int(self.args.bias_rate*len(syn_example_ids)), replace=False)
            anti_biased_example_ids = list(set(syn_example_ids) - set(biased_example_ids))
        else:
            # Eliminate the biased feature in the dev/test sets
            syn_example_ids = [i for i in range(1, len(lines))]
            biased_example_ids = []
            anti_biased_example_ids = syn_example_ids
        label_map = {label: i for i, label in enumerate(self.get_labels())}

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = None if test_mode else line[5]

                if i in syn_example_ids and not set_type.startswith("test"):
                    if i in biased_example_ids:
                        text_b = str(label_map[label]) + ' ' + text_b
                    elif i in anti_biased_example_ids:
                        noise = (int(label_map[label]) + np.random.randint(1, self.n_classes)) % self.n_classes  # Select a different class
                        text_b = str(noise) + ' ' + text_b

            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class FeverSynProcessor(DataProcessor):
    """Processor for the MultiNLI data set with synthetic bias (GLUE version)."""
    def __init__(self, args):
        self.args = args
        self.n_classes = len(self.get_labels())

    def _read_jsonlines(self, input_file):
        lines = []
        with open(input_file, "r", encoding='utf-8') as f:
            reader = jsonlines.Reader(f)
            for line in reader.iter(type=dict):
                lines.append(line)

        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(join(data_dir, "fever.train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(join(data_dir, "fever.dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type=='train':
            syn_example_ids = np.random.choice([i for i in range(1, len(lines))], size=int(self.args.synthetic_rate*len(lines)), replace=False)
            biased_example_ids = np.random.choice(syn_example_ids, size=int(self.args.bias_rate*len(syn_example_ids)), replace=False)
            anti_biased_example_ids = list(set(syn_example_ids) - set(biased_example_ids))
        else:
            # Eliminate the biased feature in the dev/test sets
            syn_example_ids = [i for i in range(1, len(lines))]
            biased_example_ids = []
            anti_biased_example_ids = syn_example_ids
        label_map = {label: i for i, label in enumerate(self.get_labels())}

        for (i, line) in enumerate(lines):
            guid = line['id']
            text_a = line['claim']
            text_b = line['evidence']
            if 'gold_label' in line:
                label = line['gold_label']
            else:
                label = line['label']
            if 'weight' in line:
                weight = line['weight']
            else:
                weight = 0.0

            if i in syn_example_ids and not set_type.startswith("test"):
                if i in biased_example_ids:
                    text_b = str(label_map[label]) + ' ' + text_b
                elif i in anti_biased_example_ids:
                    noise = (int(label_map[label]) + np.random.randint(1, self.n_classes)) % self.n_classes  # Select a different class
                    text_b = str(noise) + ' ' + text_b
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                #InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, weight=weight))
        return examples

processor_map = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mnli-syn": MnliSynProcessor,
    "mnli-mm-syn": MnliMismatchedSynProcessor,
    "hans": HansProcessor,
    "qqp": QqpProcessor,
    "qqp-syn": QqpSynProcessor,
    "paws_qqp": PawsProcessor,
    "paws_wiki": PawsProcessor,
    "fever": FeverProcessorGlue,
    "fever-syn": FeverSynProcessor,
    "sym1": FeverProcessor,
    "sym2": FeverProcessor,
}


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = processor_map[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        inputs.update({'example_id': examples[i].guid})

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class MultiDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.

    dataset_names: A list of dataset names to identify the datasets and the processors. e.g., ['mnli', 'hans']
    set_types: A list of set_type for the datasets. e.g., ['train', 'train', 'test']
    duplicates: A list of the number of times eash dataset is duplicated. e.g., [1, 2, 3]
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        processors: Optional[list] = None,
        limit_length: Optional[int] = None,
        dataset_names: Optional[list] = None,
    ):
        self.args = args
        self.output_mode = glue_output_modes[args.task_name]
        self.dataset_names = args.dataset_names.split(',') if args.dataset_names is not None else [args.task_name.lower()]
        self.processors = [processor_map[n]() for n in self.dataset_names]
        self.set_types = args.set_types.split(',') if args.set_types is not None else ['train'] * len(self.processors)
        self.duplicates = list(map(int, args.duplicates.split(','))) if args.duplicates is not None else [1] * len(self.processors)
        for dataset_name in self.dataset_names:
            assert dataset_name in processor_map, "%s is invalid, please selet from %s"%(dataset_name, list(processor_map.keys()))
        # Load data features from cache or dataset file
        cache_dataset_name = self.dataset_names if args.duplicates is None else ['%dx'%self.duplicates[i]+name for i, name in enumerate(self.dataset_names)]
        set_type_name = '_'+self.set_types[0] if len(self.set_types)==1 else ''
        cached_features_file = join(
            args.data_dir,
            #"cached_{}{}_{}_{}".format(
            #     '+'.join(cache_dataset_name), set_type_name, tokenizer.__class__.__name__, str(args.max_seq_length),
            "cached{}_{}_{}_{}".format(
                 set_type_name, tokenizer.__class__.__name__, str(args.max_seq_length), '+'.join(cache_dataset_name)
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                self.features = []
                for i, name in enumerate(self.dataset_names):
                    label_list = self.processors[i].get_labels()
                    if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                        RobertaTokenizer,
                        RobertaTokenizerFast,
                        XLMRobertaTokenizer,
                    ):
                        # HACK(label indices are swapped in RoBERTa pretrained model)
                        label_list[1], label_list[2] = label_list[2], label_list[1]
                    if name == self.args.task_name.lower():
                        name = ''
                    if self.set_types[i]=='train':
                        examples = self.processors[i].get_train_examples(join(args.data_dir, name))
                    elif self.set_types[i]=='test':
                        examples = self.processors[i].get_test_examples(join(args.data_dir, name))
                    elif self.set_types[i]=='dev':
                        examples = self.processors[i].get_dev_examples(join(args.data_dir, name))

                    examples = examples * self.duplicates[i]

                    self.features += convert_examples_to_features(
                        examples,
                        tokenizer,
                        max_length=args.max_seq_length,
                        label_list=label_list,
                        output_mode=self.output_mode,
                    )
                if limit_length is not None:
                    self.features = self.features[:limit_length]
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        dataset_name: Optional[str] = None,
        mode: Optional[str] = 'test',
    ):
        self.args = args
        self.output_mode = glue_output_modes[args.task_name]
        if dataset_name is None:
            self.dataset_name = args.task_name
            if args.synthetic_data:
                self.dataset_name = self.dataset_name + '-syn'
        else:
            self.dataset_name = dataset_name
        assert self.dataset_name in processor_map, "%s is invalid, please selet from %s"%(self.dataset_name, list(processor_map.keys()))

        if '-syn' in self.dataset_name:
            self.processor = processor_map[self.dataset_name](args)
        else:
            self.processor = processor_map[self.dataset_name]()
        # Load data features from cache or dataset file
        cached_features_file = join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                 mode, tokenizer.__class__.__name__, str(args.max_seq_length), self.dataset_name
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = self.processor.get_labels()
                #if args.task_name in ["mnli", "mnli-mm"] and dataset_name!='hans' and tokenizer.__class__ in (
                #    RobertaTokenizer,
                #    RobertaTokenizerFast,
                #    XLMRobertaTokenizer,
                #):
                #    # HACK(label indices are swapped in RoBERTa pretrained model)
                #    label_list[1], label_list[2] = label_list[2], label_list[1]

                if dataset_name is None:
                    name = ''
                else:
                    name = self.dataset_name

                if mode == 'test':
                    examples = self.processor.get_test_examples(join(args.data_dir, name))
                elif mode == 'dev':
                    examples = self.processor.get_dev_examples(join(args.data_dir, name))
                elif mode == 'train':
                    examples = self.processor.get_train_examples(join(args.data_dir, name))

                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.processor.get_labels()

