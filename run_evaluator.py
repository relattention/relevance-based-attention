from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import pandas as pd
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

from torch.nn import CrossEntropyLoss
from sklearn.utils.extmath import softmax

from tokenization_ir import BertTFIDFTokenizer
import modeling
import modeling_ir
import modeling_ir_2
import json

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class LazyDatasetClassifier(Dataset):

    def __init__(self, examples, label_list, max_seq, tokenizer, tfidf_dict):
        self.examples = examples
        self.len = len(examples)
        self.label_list = label_list
        self.max_seq = max_seq
        self.tokenizer = tokenizer
        self.tfidf_dict = tfidf_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.examples[idx]
        f = convert_example_to_features_classifier(example, self.label_list,
                                                   self.max_seq, self.tokenizer, self.tfidf_dict)
        cur_tensors = (torch.tensor(f.input_ids),
                       torch.tensor(f.input_weights),
                       torch.tensor(f.input_mask),
                       torch.tensor(f.segment_ids),
                       #torch.tensor(f.lm_label_ids),
                       torch.tensor(f.label))
        return cur_tensors


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, text_a, text_b=None, label=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label, lm_label_ids, input_weights):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.lm_label_ids = lm_label_ids
        self.input_weights = input_weights


class IRProcessor():
    """Processor for the click-through datasets."""

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, eval_set, query, documents):
        """Creates examples for the training and dev sets."""
        examples = []
        for document_id, document in documents.items():
            guid = document_id
            text_a = query
            text_b = document
            label = 0
            if query in eval_set.keys() and document_id in eval_set[query]:
                label = 1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def make_eval_set(self, data_dir, dataset_name):
        """
        input datatype 
        data_dir: str 

        output datatype
        eval_set: dict 
        key: query_token / value: list consists of clicked document_id
        documents: dict
        key: doc_id / value: list consists of document
        queries: list
        """

        evaluation_df = pd.read_csv(data_dir + dataset_name + ".tsv", sep='\t')

        queries = set()
        documents = dict()
        original_query = dict()
        for index, row in evaluation_df.iterrows():
            try:
                queries.add(row['query_token'])
                documents[row['doc_id']] = (row['title'] + row['body'])
            except:
                print("Parsing Problem :",row['title'])
                pass
            
        print(f"number of queries : {len(queries)}, number of documents : {len(documents)}")

        eval_set = dict()
        for index, row in evaluation_df.iterrows():
            query = row['query_token']
            original_query[query] = row['query']
            doc_id = row['doc_id']
            if query in eval_set:
                eval_set[query].append(doc_id)
            else:
                eval_set[query] = [doc_id]
        # we will use binary[1,0] scale for relevance rate

        # print("queries:", queries)

        return eval_set, documents, list(queries)

def convert_example_to_features_classifier(example, label_list, max_seq_length, tokenizer, tfidf_dict):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}


    query_tf_idf = dict()
    for token in example.text_a.split():
        query_tf_idf[token] = 1.0

    tokens_a, weights_a = tokenizer.tokenize(example.text_a, query_tf_idf)

    doc_tf_idf = tfidf_dict[example.guid]
    tokens_b, weights_b = tokenizer.tokenize(example.text_b, doc_tf_idf)

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, weights_a, weights_b, max_seq_length - 3)


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

    tmp = []

    for _ in weights_a:
        tmp.append(1.)

    weights = [1.] + tmp + [1.]
    weights += weights_b + [1.]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    weights += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(weights) == max_seq_length

    # logger.info("*** Example ***")
    # logger.info("guid: %s" % (example.guid))
    # logger.info("tokens: %s" % " ".join(
    #     [str(x) for x in tokens]))
    # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    # logger.info(
    #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    # logger.info("weights: %s" % " ".join([str(x) for x in weights]))
    # logger.info("Is next sentence label: %s " % (example.label))

    label_id = label_map[example.label]

    f = InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label=label_id,
                      lm_label_ids=None,
                      input_weights=weights)
    return f

def _truncate_seq_pair(tokens_a, tokens_b, w_a, w_b, max_length):
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
            w_a.pop()
        else:
            tokens_b.pop()
            w_b.pop()

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def convert_examples_to_features_for_vanilla(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair_for_vanilla(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

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
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
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

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label=label_id,
                              lm_label_ids=None,
                              input_weights=None))
    return features

def _truncate_seq_pair_for_vanilla(tokens_a, tokens_b, max_length):
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

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--dataset_name",
                        default="top300_kl", 
                        type=str, 
                        required=True, 
                        help="The name of dataset to inference (without extention ex) top300_kl)")
    parser.add_argument("--model_type",
                        default="baseline_tfidf", 
                        type=str, 
                        required=True, 
                        help="baseline, baseline_tfidf, ir-v0, ir-v1")
    parser.add_argument("--model_path",
                        default=None, 
                        type=str, 
                        required=True, 
                        help="path to model dir")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="save_path")

    ## Other parameters
    parser.add_argument("--bert_model",
                        default="bert-base-multilingual-cased",
                        type=str,
                        help="Default: bert-base-multilingual-cased" 
                         "Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--model_file",
                        default="pytorch_model.bin",
                        type=str,
                        help="The file of model (.bin), default is pytorhc_model.bin,\n" 
                             "특정 파일이 필요시 이름 설정 필요")
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    processor = IRProcessor()

    label_list = processor.get_labels()
    num_labels = len(label_list)

    print("model:", args.model_type)
    if args.model_type == "baseline": # load model (finetuned baseline on IR)
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        config = BertConfig(os.path.join(args.model_path + "bert_config.json"))
        model = BertForPreTraining(config)
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_file)))
    elif args.model_type == "baseline_tfidf": # load model (baseline_tfidf)
        tokenizer = BertTFIDFTokenizer.from_pretrained(args.bert_model, do_lower_case=False, do_tf_idf=True)
        TFIDFconfig = modeling.BertConfig(os.path.join(args.model_path + "bert_config.json"))
        model = modeling.BertTFIDFForPreTraining(TFIDFconfig)
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_file)))
    elif args.model_type == "ir-v0": # load model (*-head)
        tokenizer = BertTFIDFTokenizer.from_pretrained(args.bert_model, do_lower_case=False, do_tf_idf=True)
        head_config = modeling_ir.BertForIRConfig(os.path.join(args.model_path + "bert_config.json"))
        model = modeling_ir.BertForIRForPreTraining(head_config)
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_file)))
    elif args.model_type == "ir-v1": # load model (*-head)
        tokenizer = BertTFIDFTokenizer.from_pretrained(args.bert_model, do_lower_case=False, do_tf_idf=True)
        head_config = modeling_ir_2.BertForIRConfig(os.path.join(args.model_path + "bert_config.json"))
        model = modeling_ir_2.BertForIRForPreTraining(head_config)
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_file)))

    if args.fp16:
        model.half()
    model.to(device)

    tfidf_dict = pickle_load(os.path.join(args.data_dir, args.dataset_name + '_tfidf.pkl'))

    results_logit = dict()
    results_softmax = dict()

    eval_set, documents, queries = processor.make_eval_set(args.data_dir, args.dataset_name)
    logger.info("***** Running evaluation *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    for q_num, query in tqdm(enumerate(queries), total=len(queries), desc="Evaluating"):
    # for query in queries[0:1]: # for testing

        logger.info(f"Current Query Num : {q_num}")
        eval_examples = processor._create_examples(eval_set, query, documents)
        # logger.info("  Num examples = %d", len(eval_examples))
        if args.model_type == "baseline": # baseline or baseline_finetuned
            eval_features = convert_examples_to_features_for_vanilla(
                eval_examples, label_list, args.max_seq_length, tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Query"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    _, logits = model(input_ids, segment_ids, input_mask)

                # loss_fct = CrossEntropyLoss()
                # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                
                # eval_loss += tmp_eval_loss.mean().item()
                # nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)
        else: # baseline_tfidf or *-head model
            eval_data = LazyDatasetClassifier(eval_examples, label_list, args.max_seq_length, tokenizer, tfidf_dict)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []
                
            for batch in tqdm(eval_dataloader, desc="Query"):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_weights, input_mask, segment_ids, label_ids = batch

                with torch.no_grad():
                    _, logits = model(input_ids, input_weights, segment_ids, input_mask)
                
                # loss_fct = CrossEntropyLoss()
                # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                
                # eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)


        # eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        results_softmax[query] = []
        for i, pred in enumerate(softmax(preds)): # using softmax
            pair = dict()
            pair["score"] = pred[1]
            pair["doc_id"] = list(documents.keys())[i]
            results_softmax[query].append(pair)
        results_softmax[query].sort(reverse=True, key=lambda x: x["score"])

        ranked_doc_list = []
        for doc in results_logit[query]:
            ranked_doc_list.append(doc["doc_id"])
        results_logit[query] = ranked_doc_list

        ranked_doc_list = []
        for doc in results_softmax[query]:
            ranked_doc_list.append(doc["doc_id"])
        results_softmax[query] = ranked_doc_list

    save_name2 = args.model_path.split('/')[0] + '_' + args.model_file.split('.')[0] \
                 + '_' + args.dataset_name + '_output.json'
    path2 = os.path.join(args.output_dir,
                         save_name2)

    with open(path2, 'w', encoding="utf8") as f:
        json.dump(results_softmax, f, indent=4, sort_keys=True, ensure_ascii=False)
        # print("preds:", preds[0:10])
        # print("preds softmax:", softmax(preds)[0:10])


if __name__ == "__main__":
    main()
