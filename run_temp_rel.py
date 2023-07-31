"""
This code is based on the file in SpanBERT repo: https://github.com/facebookresearch/SpanBERT/blob/master/code/run_tacred.py
"""

#import argparse
import logging
import os
import random
import time
import json
import sys
import pickle
import numpy as np
import torch
import re
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from relation.models import BertForRelation, AlbertForRelation
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from evaluate.closure import evaluation as closure_evaluate

from relation.utils import generate_temp_relation_data, decode_sample_id
from custom.const import task_rel_labels, task_ner_labels
from helper_functions import set_arguments

#from timegraph.i2b2Evaluation import get_i2b2_results#(goldDir, systemDir

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx
        

def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>'%label)
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d'%len(tokenizer))

def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, args, unused_tokens=True, type_file="dev", load_pred_after_training=False, load_pred_after_training_file=None):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    
    ** the subj_start/end are prepared to be at corresponding token level
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
                # NOTE Unused tokens are helpful if you want to introduce specific words to your fine-tuning
                # or further pre-training procedure; they allow you to treat words that are relevant only 
                # in your context just like you want, and avoid subword splitting that would occur with 
                # the original vocabulary of BERT.
                # https: // stackoverflow.com/questions/62452271/understanding-bert-vocab-unusedxxx-tokens
                # NOTE: for our case I think the reserved num of reserved unused tokens are within limit
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        # REVIEW: check out the special token_tokens list
        return special_tokens[w] # I'm stunned they all have and unused%d

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    features = []
    
    # NOTE: the only variables to change here seems to be marker, model, and context_window
    if not load_pred_after_training:
        file_name = f"cached_{type_file}_{args.task}_{args.marker}_{args.model}_{args.context_window}.pkl"
        #file_path = os.path.join(args.output_dir, file_name)
        rel_level_dir = "/".join(args.output_dir.split('/')[:-1])
        file_path = os.path.join(rel_level_dir, file_name)
    else:
        file_path = load_pred_after_training_file  
         
    if load_pred_after_training  or (os.path.exists(file_path) and args.use_cached):
        with open(file_path, "rb") as f:
            features = pickle.load(f)
        print(f"**Exists: {file_path}, loaded")
    else:    
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = [CLS]

            SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])
            SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])
            OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
            OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])


            """"
            # DEBUG: 
            by debugging observation, these subj_start/obj_start are apparently sentence level
            """
            for i, token in enumerate(example['token']):
                if i == example['subj_start']:
                    sub_idx = len(tokens)
                    tokens.append(SUBJECT_START_NER)
                if i == example['obj_start']:
                    obj_idx = len(tokens)
                    tokens.append(OBJECT_START_NER)
                for sub_token in tokenizer.tokenize(token): # word-piece happening here
                    tokens.append(sub_token)
                if i == example['subj_end']:
                    tokens.append(SUBJECT_END_NER)
                if i == example['obj_end']:
                    tokens.append(OBJECT_END_NER)
            tokens.append(SEP)

            num_tokens += len(tokens)
            max_tokens = max(max_tokens, len(tokens))

            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                if sub_idx >= max_seq_length:
                    sub_idx = 0
                if obj_idx >= max_seq_length:
                    obj_idx = 0
            else:
                num_fit_examples += 1
            
            # NOTE: all words have been correctedly tokenized here using word-piece in **tokens**
            # NOTE: padding and further process it down below
            # NOTE: fix length padding is slow, consider dynamic padding which improves speed: https://mccormickml.com/2020/07/29/smart-batching-tutorial/
            # TODO: https://huggingface.co/course/chapter7/2?fw=pt
            # REVIEW: consider writing a hugging face collector to speed things up
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids)) 
            input_ids += padding
            input_mask += padding
            segment_ids += padding # NOTE: Segment id seems to be all zeros, what's the use? - # token_type_ids is the same as the "segment ids", which differentiates sentence 1 and 2 in 2-sentence tasks
            # KeyError: '' docid=256, <TLINK id="TL60" fromID="E52" fromText="his blood pressure" toID="E49" toText="his blood pressure" type="AFTER" />  
            # KeyError: '' docid=318, <TLINK id="TL0" fromID="E0" fromText="ADMISSION" toID="T0" toText="8/21/93" type="OVERLAP" />
                        # <TLINK id="TL1" fromID="E7" fromText="DISCHARGE" toID="T3" toText="8/23/93" type="OVERLAP" />       
            label_id = label2id[example['relation']]
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length


            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id,
                                sub_idx=sub_idx,
                                obj_idx=obj_idx))
        
        with open(file_path, "wb") as f:
            pickle.dump(features, f)
        print(f"**Saved: {file_path} ")
    
    if not load_pred_after_training:
        # NOTE: the info below will only be printed if file does not exists before
        logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
        logger.info("Max #tokens: %d"%max_tokens)
        logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                    num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features


def convert_examples_to_features_order_swapped(examples, label2id, max_seq_length, tokenizer, special_tokens, args, unused_tokens=True, type_file="dev", load_pred_after_training=False, load_pred_after_training_file=None):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    
    ** the subj_start/end are prepared to be at corresponding token level
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
                # NOTE Unused tokens are helpful if you want to introduce specific words to your fine-tuning
                # or further pre-training procedure; they allow you to treat words that are relevant only 
                # in your context just like you want, and avoid subword splitting that would occur with 
                # the original vocabulary of BERT.
                # https: // stackoverflow.com/questions/62452271/understanding-bert-vocab-unusedxxx-tokens
                # NOTE: for our case I think the reserved num of reserved unused tokens are within limit
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        # REVIEW: check out the special token_tokens list
        return special_tokens[w] # I'm stunned they all have and unused%d

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    features = []
    
    # NOTE: the only variables to change here seems to be marker, model, and context_window
    if not load_pred_after_training:
        file_name = f"flipped_cached_{type_file}_{args.task}_{args.marker}_{args.model}_{args.context_window}.pkl"
        #file_path = os.path.join(args.output_dir, file_name)
        rel_level_dir = "/".join(args.output_dir.split('/')[:-1])
        file_path = os.path.join(rel_level_dir, file_name)
    else:
        file_path = load_pred_after_training_file  
         
    if load_pred_after_training  or (os.path.exists(file_path) and args.use_cached):
        with open(file_path, "rb") as f:
            features = pickle.load(f)
        print(f"**Exists: {file_path}, loaded")
    else:    
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = [CLS]

            # SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])
            # SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])
            # OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
            # OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])
            SUBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
            SUBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])
            OBJECT_START_NER =  get_special_token("SUBJ_START=%s"%example['subj_type'])
            OBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])


            """"
            # DEBUG: 
            by debugging observation, these subj_start/obj_start are apparently sentence level
            """
            # DEBUG: order swapped
            for i, token in enumerate(example['token']):
                if i == example['obj_start']:
                    sub_idx = len(tokens)
                    tokens.append(SUBJECT_START_NER)
                if i == example['subj_start']:
                    obj_idx = len(tokens)
                    tokens.append(OBJECT_START_NER)
                for sub_token in tokenizer.tokenize(token): # word-piece happening here
                    tokens.append(sub_token)
                if i == example['obj_end']:
                    tokens.append(SUBJECT_END_NER)
                if i == example['subj_end']:
                    tokens.append(OBJECT_END_NER)
            tokens.append(SEP)

            num_tokens += len(tokens)
            max_tokens = max(max_tokens, len(tokens))

            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                if sub_idx >= max_seq_length:
                    sub_idx = 0
                if obj_idx >= max_seq_length:
                    obj_idx = 0
            else:
                num_fit_examples += 1
            
            # NOTE: all words have been correctedly tokenized here using word-piece in **tokens**
            # NOTE: padding and further process it down below
            # NOTE: fix length padding is slow, consider dynamic padding which improves speed: https://mccormickml.com/2020/07/29/smart-batching-tutorial/
            # TODO: https://huggingface.co/course/chapter7/2?fw=pt
            # REVIEW: consider writing a hugging face collector to speed things up
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids)) 
            input_ids += padding
            input_mask += padding
            segment_ids += padding # NOTE: Segment id seems to be all zeros, what's the use? - # token_type_ids is the same as the "segment ids", which differentiates sentence 1 and 2 in 2-sentence tasks
            # KeyError: '' docid=256, <TLINK id="TL60" fromID="E52" fromText="his blood pressure" toID="E49" toText="his blood pressure" type="AFTER" />  
            # KeyError: '' docid=318, <TLINK id="TL0" fromID="E0" fromText="ADMISSION" toID="T0" toText="8/21/93" type="OVERLAP" />
                        # <TLINK id="TL1" fromID="E7" fromText="DISCHARGE" toID="T3" toText="8/23/93" type="OVERLAP" />       
            def swap_label_order(l):
                if l=="BEFORE":
                    return "AFTER"
                if l=="AFTER":
                    return "BEFORE"
                return l
            label_id = swap_label_order(label2id[example['relation']])
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length


            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id,
                                sub_idx=sub_idx,
                                obj_idx=obj_idx))
        
        with open(file_path, "wb") as f:
            pickle.dump(features, f)
        print(f"**Saved: {file_path} ")
    
    if not load_pred_after_training:
        # NOTE: the info below will only be printed if file does not exists before
        logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
        logger.info("Max #tokens: %d"%max_tokens)
        logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                    num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_f1(preds, labels, e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

        if e2e_ngold is not None:
            e2e_recall = n_correct * 1.0 / e2e_ngold
            e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall)
        else:
            e2e_recall = e2e_f1 = 0.0
        # Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?
        return {'precision': prec, 'recall': e2e_recall, 'f1': e2e_f1, 'task_recall': recall, 'task_f1': f1, 
        'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': e2e_ngold, 'task_ngold': n_gold}


def evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=None, verbose=True, test_examples=None, args=None, id2label=None, eval_type="dev"):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    logits = preds[0]
    preds = np.argmax(preds[0], axis=1)
    result = compute_f1(preds, eval_label_ids.numpy(), e2e_ngold=e2e_ngold)
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    
    # NOTE: write model output to predicted_xml_dir

    predicted_xml_dir = os.path.join(args.output_dir, f"predicted_xml_{eval_type}")
    if not os.path.exists(predicted_xml_dir):
        os.mkdir(predicted_xml_dir)
    
    doc_link_pred = {}
    doc_link_prob = {}
    #probs = get_probs(data)
    #assert len(probs)==len(pred_labels)
    #/data/chengc7/MarkerTRel/preprocess/corpus/i2b2/train_merged
    test_xml_dir = ""
    if eval_type=="dev":
        test_xml_dir = "/data/chengc7/MarkerTRel/preprocess/corpus/i2b2/dev_xml"
    else:
        test_xml_dir = "/data/chengc7/MarkerTRel/preprocess/corpus/i2b2/ground_truth/merged_xml"
        
    pred_labels = [id2label[p] for p in preds]
    probs = [0.0 for p in preds]
    for (ex, pred_l, prob_l)in zip(test_examples, pred_labels, probs): # unaugmented_examples
        if ex['docid'] not in doc_link_pred:
            doc_link_pred[ex['docid']] = {}
            doc_link_prob[ex['docid']] = {}
        doc_link_pred[ex['docid']][ex['lid']]=pred_l 
        doc_link_prob[ex['docid']][ex['lid']]=prob_l
    for doc_id in doc_link_pred.keys():
        ce = closure_evaluate(doc_id, doc_link_pred[doc_id], doc_link_prob[doc_id], test_xml_dir, predicted_xml_dir)
        ce.eval()
        res_keys = ['Precicion', 'Recall', 'Average P&R', 'F measure']
        res_dict = {}
    
    
    result_file = os.path.join(args.output_dir, f"result_{eval_type}.txt")
    os.system(' '.join(["python2 /data/chengc7/MarkerTRel/evaluate/i2b2Evaluation.py --tempeval", test_xml_dir, predicted_xml_dir]) + ' > ' + result_file)
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            lines = f.readlines()
            res_idx = 6
            #print(lines[-res_idx:])
            # res = lines[-res_idx:]
            res_keys = ['Precicion', 'Recall', 'Average P&R', 'F measure']
            res_dict = {}
            
            for i, (k, line)in enumerate(zip(res_keys, lines[-res_idx:])):
                if k!='Average P&R':
                    m = re.search(r".*\t(.*)\n.*", line)
                    val = m.group(1)
                    res_dict[k] = val
    
                    result[f'i2b2_{k}'] = float(res_dict[k])
    result['i2b2_f1'] = result['i2b2_F measure']
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result, logits


def evaluate_test_swap_order(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=None, verbose=True):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    logits = preds[0]
    preds = np.argmax(preds[0], axis=1)
    result = compute_f1(preds, eval_label_ids.numpy(), e2e_ngold=e2e_ngold)
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result, logits


def print_pred_json(eval_data, eval_examples, preds, id2label, output_file, pred_probs=None):
    # debug
    """
    eval_data: small one,
    eval_examples: augmented one, 
    preds: augmented one
    pred_probs added
    """
    ## 
    rels = dict()
    probs = dict()
    for ex, pred, prob in zip(eval_examples, preds, pred_probs):
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if doc_sent not in probs:
            probs[doc_sent] = []
        
        # NOTE: C1 and C2
        #C1 ALWAYS BE TRUE, ALWAYS 1, 2, 3 for i2b2
        # C2  make this only save original (non-augmented) tlinks 
        if (pred != 0) and ('flipped' not in ex['lid']): 
            rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], id2label[pred]])
            probs[doc_sent].append([sub[0], sub[1], obj[0], obj[1], str(prob)]) # str(prob) for json serialization


    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        doc['prob'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))
            doc['prob'].append(probs.get(k, []))

    # if not test_order_swap:
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))
    # else:
    #     logger.info('Output predictions to %s..'%(output_file + ))
    #     with open(output_file, 'w') as f:
    #         f.write('\n'.join(json.dumps(doc) for doc in js))
    # else:

def setseed(seed):
    random.seed(seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_trained_model(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s'%output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def main(args):
    if 'albert' in args.model:
        RelationModel = AlbertForRelation
        args.add_new_tokens = True
    else:
        # TODO: why not also add new tokens for bert as in set args.add_marker_tokens=True? may be dealt in bash file
        RelationModel = BertForRelation

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

# DATA tr/te/eval datasets, later translated to TensorDataset, and Dataloader
    # train set
    if args.do_train:
        train_dataset, train_examples, train_nrel_original = generate_temp_relation_data(os.path.join(args.data_dir, "train.json"), entity_type=args.marker, use_gold=True, context_window=args.context_window)
    # dev set
    if (args.do_eval and args.do_train) or (args.do_eval and not(args.eval_test)):
        eval_dataset, eval_examples, eval_nrel_original = generate_temp_relation_data(os.path.join(args.data_dir, "dev.json"), entity_type=args.marker, use_gold=args.eval_with_gold, context_window=args.context_window)
    # test set
    if args.eval_test:
        test_dataset, test_examples, test_nrel_original = generate_temp_relation_data(os.path.join(args.data_dir, "test.json"), entity_type=args.marker, use_gold=args.eval_with_gold, context_window=args.context_window)

    setseed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(
        device, n_gpu))

    # get label_list
    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # NOTE: tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    if args.add_new_tokens:

        # NOTE:  has to resize token embedding after add marker tokens (it's in code later) 
        add_marker_tokens(tokenizer, task_ner_labels[args.marker])

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}
    
    
    if args.do_eval and (args.do_train or not(args.eval_test)):
        # DATA: data - eval
        # TODO undestand what's utilized here
        # docid=256, <TLINK id="TL60" 
        #examples, label2id, max_seq_length, tokenizer, special_tokens, args, unused_tokens=True, type_file="dev"
        eval_features = convert_examples_to_features(
            eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.add_new_tokens), args=args, type_file="dev")
        eval_nrel = len(eval_features)
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
        # DATA: eval dataloader
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids
    # REVIEW: what's special tokens?
    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)
    if args.do_train:
        # DATA: tr data to features
        train_features = convert_examples_to_features(
            train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.add_new_tokens), args=args, type_file="train")
        train_nrel = len(train_features)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in train_features], dtype=torch.long)
        # DATA: tr dataset and dataloader 
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        
        lr = args.learning_rate
        
        # NOTE: model
        model = RelationModel.from_pretrained(
            args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)
        model.bert.resize_token_embeddings(len(tokenizer))
        # if hasattr(model, 'bert'):
        #     # NOTE: NEED to resize tokenizer
        #     model.bert.resize_token_embeddings(len(tokenizer))
        # elif hasattr(model, 'albert'):
        #     model.albert.resize_token_embeddings(len(tokenizer))
        # else:
        #     raise TypeError("Unknown model class")

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not(args.bertadam))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids, sub_idx, obj_idx)
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % eval_step == 0:
                    logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps))
                    #save_model = False
                    if args.do_eval:
                        #def evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=None, verbose=True, test_examples=None, args=None, id2label=None):
    
                        preds, result, logits = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel, verbose=True, test_examples=eval_examples, args=args, id2label=id2label, eval_type="dev") # def evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=None, verbose=True, test_examples=None, args=None, id2label=None):

                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.train_batch_size

                        if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                            best_result = result
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                            save_trained_model(args.output_dir, model, tokenizer)

    evaluation_results = {}
    if args.do_eval:
        logger.info(special_tokens)
        if args.eval_test:
            # DATA - prep eval data
            eval_dataset = test_dataset
            eval_examples = test_examples
            
            # debug: added for test with swapped order
            if not args.test_order_swap:
                eval_features = convert_examples_to_features(
                    test_examples, label2id, args.max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.add_new_tokens),args=args, type_file="test")
            else:
                eval_features = convert_examples_to_features_order_swapped(test_examples, label2id, args.max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.add_new_tokens),args=args, type_file="test")

            eval_nrel = len(eval_features)
            logger.info(special_tokens)
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
            all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
            eval_label_ids = all_label_ids
        
        # NOTE: model
        model = RelationModel.from_pretrained(args.output_dir, num_rel_labels=num_labels)
        model.to(device)
        #preds, result, logits = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel, verbose=True, test_examples=eval_examples, args=args, id2label=id2label)
        preds, result, logits = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel, verbose=True, test_examples=test_examples, args=args, id2label=id2label, eval_type="test")

        # if args.test_order_swap:
        #     convert_examples_to_features_order_swapped(examples, label2id, max_seq_length, tokenizer, special_tokens, args, unused_tokens=True, type_file="dev", load_pred_after_training=False, load_pred_after_training_file=None)

        logger.info('*** Evaluation Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        #  DEBUG: THIS HAVE THE AUGMENTED ONES OVERWRITTEN IN WERID WAYS
        # logits = preds[0] # Bx num_labels
        # preds = np.argmax(logits, axis=1) 
        # logits.shape: (27654, 4), label_list: ['no_relation', 'BEFORE', 'AFTER', 'OVERLAP']
        softmax = torch.nn.Softmax()
        pred_probs = np.max(softmax(torch.tensor(logits)).numpy(), axis=1)
        assert len(pred_probs)==len(preds)
        if not args.test_order_swap:
            print_pred_json(eval_dataset, eval_examples, preds, id2label, os.path.join(args.output_dir, args.prediction_file), pred_probs)
        else:
            print_pred_json(eval_dataset, eval_examples, preds, id2label, os.path.join(args.output_dir, 'flipped_'+args.prediction_file), pred_probs)

if __name__ == "__main__":
    
    args = set_arguments()
    main(args)
