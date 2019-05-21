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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open

import pandas as pd
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from modeling import BertForPreTraining, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LazyDataset(Dataset):

    def __init__(self, examples, max_seq, tokenizer):
        self.examples = examples
        self.len = len(examples)
        self.max_seq = max_seq
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.examples[idx]
        f = convert_example_to_features(example, self.max_seq, self.tokenizer)
        cur_tensors = (torch.tensor(f.input_ids),
                       torch.tensor(f.input_mask),
                       torch.tensor(f.segment_ids),
                       torch.tensor(f.lm_label_ids),
                       torch.tensor(f.label))
        return cur_tensors


class LazyDatasetClassifier(Dataset):

    def __init__(self, examples, label_list, max_seq, tokenizer):
        self.examples = examples
        self.len = len(examples)
        self.label_list = label_list
        self.max_seq = max_seq
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.examples[idx]
        f = convert_example_to_features_classifier(example, self.label_list, self.max_seq, self.tokenizer)
        cur_tensors = (torch.tensor(f.input_ids),
                       torch.tensor(f.input_mask),
                       torch.tensor(f.segment_ids),
                       # torch.tensor(f.lm_label_ids),
                       torch.tensor(f.label))
        return cur_tensors


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        train = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t')

        with open(os.path.join(data_dir, 'doc.pkl'), 'rb') as f:
            doc = pickle.load(f)

        train = train[train['query'].map(lambda x: isinstance(x, str))]

        return train.apply(lambda x: InputExample(guid=x['doc_id'],
                                                  text_a=x['query'],
                                                  text_b=doc[x['doc_id']],
                                                  label=x['click'],
                                                  lm_labels=None), axis=1)

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""

        dev = pd.read_csv(os.path.join(data_dir, 'dev.tsv'), sep='\t')

        with open(os.path.join(data_dir, 'doc.pkl'), 'rb') as f:
            doc = pickle.load(f)

        dev = dev[dev['query'].map(lambda x: isinstance(x, str))]

        return dev.apply(lambda x: InputExample(guid=x['doc_id'],
                                                text_a=x['query'],
                                                text_b=doc[x['doc_id']],
                                                label=x['click'],
                                                lm_labels=None), axis=1)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [0, 1]


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, text_a, text_b=None, label=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, b ut not for test examples.
            lm_labels: Masked LM 레이블
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label  # click 여부
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features_classifier(example, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    tokens_b = tokenizer.tokenize(example.text_b)
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    tokens += tokens_b + ["[SEP]"]
    segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    f = InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label=label_id,
                      lm_label_ids=None)
    return f


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    features = []
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = tokenizer.tokenize(example.text_b)
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    # if ex_index < 5:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("tokens: %s" % " ".join(
    #         [str(x) for x in tokens]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     logger.info(
    #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logger.info("LM label: %s " % (lm_label_ids))
    #     logger.info("Is next sentence label: %s " % (example.is_next))

    f = InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      lm_label_ids=lm_label_ids,
                      label=example.label)
    return f


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj=obj, file=f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--bert_model", default='bert-base-multilingual-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    # parser.add_argument("--do_eval",
    #                     action='store_true',
    #                     help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for training.")
#     parser.add_argument("--eval_batch_size",
#                         default=2,
#                         type=int,
#                         help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on GPUs")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--visdom', action='store_true',
                        help='Use visdom for loss visualization')
    parser.add_argument('--check_saved_model', action='store_true',
                        help='Use visdom for loss visualization')
    parser.add_argument('--last_final_epoch', type=int, default=-1,
                        help="저번에 이미 최종 학습을 했고, 이에 이어서 트레이닝을 원할때 사용,\n"
                             "기존에 train_epoch를 3으로 세팅했다면, 2가 아닌 3을 입력하세요.")

    args = parser.parse_args()
    print(args)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
        # visdom을 통해서 loss를 시각화

    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)

    processor = DataProcessor()
    label_list = processor.get_labels()

    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.data_dir)

        train_examples = processor.get_train_examples(args.data_dir)
        train_dataset = LazyDataset(train_examples, args.max_seq_length, tokenizer)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)

        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    loaded_epoch = -1
    saved_model_path = -1

    if args.last_final_epoch != -1:
        last_model = os.path.join(args.output_dir, WEIGHTS_NAME)
        if os.path.exists(last_model):
            saved_model_path = last_model
            loaded_epoch = args.last_final_epoch - 1

    elif args.check_saved_model:
        for epoch in range(int(args.num_train_epochs)):
            tmp = os.path.join(args.output_dir, (f"weight_on_ep{epoch}_" + WEIGHTS_NAME))
            if os.path.exists(tmp):
                saved_model_path = tmp
                loaded_epoch = epoch

    if saved_model_path != -1:
        logger.info(f"Loading on saved model {saved_model_path}")
        config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(config_file)
        logger.info("Model config {}".format(config))
        model = BertForPreTraining(config)
        model.load_state_dict(torch.load(saved_model_path))
    else:
        loaded_epoch = -1
        model = BertForPreTraining.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.visdom:
        # 일단 visdom 기본 figure를 정의
        vis_title = f'Baseline on {len(train_dataset)} dataset'
        vis_legend = ['LM Loss', 'Click Loss', 'Total Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(viz, 'Epoch', 'Loss', vis_title, vis_legend)

    # if args.do_eval:
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #
    #     eval_data = LazyDatasetClassifier(eval_examples, label_list, args.max_seq_length, tokenizer)
    #     # Run prediction for full data
    #     """
    #     cur_tensors = (torch.tensor(f.input_ids),
    #            torch.tensor(f.input_mask),
    #            torch.tensor(f.segment_ids),
    #            torch.tensor(f.lm_label_ids),
    #            torch.tensor(f.label))
    #     """
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #     save_eval_loss = []

    global_step = 0
    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        """
        cur_tensors = (torch.tensor(f.input_ids),
               torch.tensor(f.input_mask),
               torch.tensor(f.segment_ids),
               torch.tensor(f.lm_label_ids),
               torch.tensor(f.label))
        """

        save_loss = []
        save_epoch_loss = []
        save_step = int(len(train_dataloader) // 5)

        for epoch in trange((loaded_epoch + 1), int(args.num_train_epochs), desc="Epoch"):

            #     if args.do_eval and loaded_epoch != -1:
            #         model.eval()
            #         eval_loss, eval_accuracy = 0, 0
            #         nb_eval_steps, nb_eval_examples = 0, 0
            #
            #         for batch in tqdm(eval_dataloader, desc="Evaluating"):
            #             batch = tuple(t.to(device) for t in batch)
            #             input_ids, input_mask, segment_ids, label_ids = batch
            #
            #             with torch.no_grad():
            #                 tmp_eval_loss = model(input_ids, segment_ids, input_mask, None, label_ids)
            #                 prediction_scores, logits = model(input_ids, segment_ids, input_mask)
            #
            #             if n_gpu > 1:
            #                 tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu.
            #
            #             logits = logits.detach().cpu().numpy()
            #             label_ids = label_ids.to('cpu').numpy()
            #             tmp_eval_accuracy = accuracy(logits, label_ids)
            #
            #             eval_loss += tmp_eval_loss.mean().item()
            #             eval_accuracy += tmp_eval_accuracy
            #
            #             nb_eval_examples += input_ids.size(0)
            #             nb_eval_steps += 1
            #
            #         eval_loss = eval_loss / nb_eval_steps
            #         eval_accuracy = eval_accuracy / nb_eval_examples
            #         result = {'eval_loss': eval_loss,
            #                   'eval_accuracy': eval_accuracy,
            #                   'global_step': global_step}
            #
            #         save_eval_loss.append(eval_loss)
            #
            #         output_eval_file = os.path.join(args.output_dir, f"Epoch_{epoch}_eval_results.txt")
            #         with open(output_eval_file, "w") as writer:
            #             logger.info(f"***** Eval results on Epoch {epoch} *****")
            #             for key in sorted(result.keys()):
            #                 logger.info("  %s = %s", key, str(result[key]))
            #                 writer.write("%s = %s\n" % (key, str(result[key])))

            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_loss_ml = 0
            tr_loss_click = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, label = batch
                # if global_step == 0:
                #     print(input_ids.shape, input_mask.shape, segment_ids.shape, lm_label_ids.shape, label.shape)
                loss, loss_ml, loss_click = model(input_ids, segment_ids, input_mask, lm_label_ids, label)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    loss_ml = loss_ml.mean()
                    loss_click = loss_click.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_ml = loss_ml / args.gradient_accumulation_steps
                    loss_click = loss_click / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                tr_loss_ml += loss_ml.item()
                tr_loss_click += loss_click.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step != 0 and global_step % save_step == 0:
                    # 한 에포치당 5번 저장
                    logger.info(f'Saving state, iter: {global_step}')
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # Only save the model it-self
                    model_name = f"weight_on_{global_step}_" + WEIGHTS_NAME
                    output_model_file = os.path.join(args.output_dir, model_name)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())
                    print("Loss at ", global_step, loss_ml.item(), loss_click.item(), loss.item())

                save_loss.append([loss_ml.item(), loss_click.item(), loss.item()])

                if args.visdom:
                    update_vis_plot(viz, global_step, loss_ml.item(), loss_click.item(),
                                    iter_plot, epoch_plot, 'append')

            if epoch != (int(args.num_train_epochs) - 1):
                # 각 에포치가 끝날때 마다 저장
                logger.info(f'Saving state, epoch: {epoch}')
                model_to_save = model.module if hasattr(model, 'module') else model
                # Only save the model it-self
                model_name = f"weight_on_ep{epoch}_" + WEIGHTS_NAME
                output_model_file = os.path.join(args.output_dir, model_name)
                torch.save(model_to_save.state_dict(), output_model_file)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                print("Loss at epoch", epoch, tr_loss_ml, tr_loss_click, tr_loss)

            save_epoch_loss.append([tr_loss_ml, tr_loss_click, tr_loss])
            if args.visdom:
                update_vis_plot(viz, epoch, tr_loss_ml, tr_loss_click,
                                epoch_plot, None, 'append', len(train_dataset) // args.train_batch_size)

            # if args.do_eval and loaded_epoch == -1:
            #
            #     model.eval()
            #     eval_loss, eval_accuracy = 0, 0
            #     nb_eval_steps, nb_eval_examples = 0, 0
            #
            #     for batch in tqdm(eval_dataloader, desc="Evaluating"):
            #         batch = tuple(t.to(device) for t in batch)
            #         input_ids, input_mask, segment_ids, label_ids = batch
            #
            #         with torch.no_grad():
            #             tmp_eval_loss = model(input_ids, segment_ids, input_mask, None, label_ids)
            #             prediction_scores, logits = model(input_ids, segment_ids, input_mask)
            #
            #         if n_gpu > 1:
            #             tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu.
            #
            #         logits = logits.detach().cpu().numpy()
            #         label_ids = label_ids.to('cpu').numpy()
            #         tmp_eval_accuracy = accuracy(logits, label_ids)
            #
            #         eval_loss += tmp_eval_loss.mean().item()
            #         eval_accuracy += tmp_eval_accuracy
            #
            #         nb_eval_examples += input_ids.size(0)
            #         nb_eval_steps += 1
            #
            #     eval_loss = eval_loss / nb_eval_steps
            #     eval_accuracy = eval_accuracy / nb_eval_examples
            #     result = {'eval_loss': eval_loss,
            #               'eval_accuracy': eval_accuracy,
            #               'global_step': global_step}
            #
            #     save_eval_loss.append(eval_loss)
            #
            #     output_eval_file = os.path.join(args.output_dir, f"Epoch_{epoch}_eval_results.txt")
            #     with open(output_eval_file, "w") as writer:
            #         logger.info(f"***** Eval results on Epoch {epoch} *****")
            #         for key in sorted(result.keys()):
            #             logger.info("  %s = %s", key, str(result[key]))
            #             writer.write("%s = %s\n" % (key, str(result[key])))

        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        # if args.do_train:
        #     torch.save(model_to_save.state_dict(), output_model_file)

        save_loss = np.array(save_loss)
        save_epoch_loss = np.array(save_epoch_loss)
        np.save(os.path.join(args.output_dir, "save_loss.npy"), save_loss)
        np.save(os.path.join(args.output_dir, "save_epoch_loss.npy"), save_epoch_loss)

        # if args.do_eval:
        #     save_eval_loss = np.array(save_eval_loss)
        #     np.save(os.path.join(args.output_dir, "save_eval_loss.npy"), save_eval_loss)

        model_to_save = model.module if hasattr(model, 'module') else model
        # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # # Load a trained model and config that you have fine-tuned
        # config = BertConfig(output_config_file)
        # model = BertForPreTraining(config)
        # model.load_state_dict(torch.load(output_model_file))
        # model.to(device)
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0 and window2 is not None:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
