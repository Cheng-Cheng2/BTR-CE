import json
import logging
import sys
import functools
import random
import os
sys.path.append('/data/chengc7/MarkerTRel')
from custom.data_structures import Dataset

logger = logging.getLogger('root')

def decode_sample_id(sample_id):
    # sample['docid'] = doc._doc_key
    # sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
    doc_sent = sample_id.split('::')[0] #'%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix
    pair = sample_id.split('::')[1] # (%d,%d)-(%d,%d)'%(sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))

    return doc_sent, sub, obj


#
def generate_relation_data(entity_data, use_gold=True, context_window=0):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    logger.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data, entity_type="relations") # NOTE: self.documents contains lst of Documents

    nner, nrel = 0, 0
    max_sentsample = 0
    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            # NOTE: for each sent in each doc
            sent_samples = []

            nner += len(sent.ner)
            nrel += len(sent.relations)
            # if use_gold:
            sent_ner = sent.ner
            # else:
            #     sent_ner = sent.predicted_ner
            
            # DEBUG - the two gold dcts did not same to be sued any wher 
             
            # gold_ner = {} # ner.span:ner.label-text
            # for ner in sent.ner:
            #     gold_ner[ner.span] = ner.label
            
            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label
            # DEBUG: sent_start is always 0 here, it's only used for adjust context?
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            # TODO: understand context
            if context_window > 0:
                add_left = (context_window-len(sent.text)) // 2
                add_right = (context_window-len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1
            
            # NOTE: this is how subject, object label are created? 
            # # TODO when do this for ours, should not create sub, obj relation pair for each x, y of NER
            # for every pair of named entities,
            for x in range(len(sent_ner)):          
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = gold_rel.get((sub.span, obj.span), 'no_relation') # NOTE: if k does not exist set to 'no_relation'
                   
                    # NOTE: only create sub/obj pair if relation exists
                    if label != 'no_relation':
                        sample = {}
                        sample['docid'] = doc._doc_key
                        sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                        sample['relation'] = label
                        
                        # NOTE: sent_start is only used for adjusting context, these are sent_level: subj_start, sub_end, obj_start, obj_end
                            # NOTE: they are different from sub.span.start_doc/end_doc... (printed above)
                        sample['subj_start'] = sub.span.start_sent + sent_start # 
                        sample['subj_end'] = sub.span.end_sent + sent_start# 
                        sample['subj_type'] = sub.label
                        sample['obj_start'] = obj.span.start_sent + sent_start
                        sample['obj_end'] = obj.span.end_sent + sent_start
                        sample['obj_type'] = obj.label
                        sample['token'] = tokens
                        sample['sent_start'] = sent_start
                        sample['sent_end'] = sent_end

                        sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    return data, samples, nrel


def set_sample(sub, obj, label, sent_start, sent_end, tokens, doc_id, lid, sentence_ix):
    sample = {}
    sample['docid'] = doc_id
    # sample['docid'] = doc._doc_key
    # sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc_id, sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
    sample['lid'] = lid
    sample['relation'] = label
    
    # NOTE: sent_start is only used for adjusting context, these are sent_level: subj_start, sub_end, obj_start, obj_end
        # NOTE: they are different from sub.span.start_doc/end_doc... (printed above)
    sample['subj_start'] = sub.span.start_sent + sent_start # 
    sample['subj_end'] = sub.span.end_sent + sent_start# 
    sample['subj_type'] = sub.label
    sample['obj_start'] = obj.span.start_sent + sent_start
    sample['obj_end'] = obj.span.end_sent + sent_start
    sample['obj_type'] = obj.label
    sample['token'] = tokens
    sample['sent_start'] = sent_start
    sample['sent_end'] = sent_end
    return sample

def generate_temp_relation_data(entity_data, entity_type, use_gold=True, context_window=0, task="i2b2", max_seq_length=512): # DEBUG: added task for tbd not flipping
    """
    Prepare data for the relation model
    If training: set use_gold = True
    
    
    return: list of sample: (id, relation)
                sample['docid'] = doc._doc_key
                        sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                        sample['relation'] = label
                        
                        # NOTE: sent_start is only used for adjusting context, these are sent_level: subj_start, sub_end, obj_start, obj_end
                            # NOTE: they are different from sub.span.start_doc/end_doc... (printed above)
                        sample['subj_start'] = sub.span.start_sent + sent_start # 
                        sample['subj_end'] = sub.span.end_sent + sent_start# 
                        sample['subj_type'] = sub.label
                        sample['obj_start'] = obj.span.start_sent + sent_start
                        sample['obj_end'] = obj.span.end_sent + sent_start
                        sample['obj_type'] = obj.label
                        sample['token'] = tokens
                        sample['sent_start'] = sent_start
                        sample['sent_end'] = sent_end
    """
    logger.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data, entity_type=entity_type, context_window=context_window) # NOTE: self.documents contains lst of Documents

    nner, nrel = 0, 0
    max_sentsample = 0
    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            # NOTE: for each sent in each doc
            sent_samples = []
            nner += len(sent.ner)
            nrel += len(sent.relations)
            
            # TODO: sent_start is always 0 here unless context_window>0, it's only used for adjust context?
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text
            gold_rel = sent.relations[0]
            # NOTE: create sample, and add flipped sample for before/after
            sub, obj = sent.ner
            label = gold_rel.label # NOTE: if k does not exist set to 'no_relation'
            # NOTE: only create sub/obj pair if relation exists
            # TODO: understand context
            # if context_window > 0:
            #     add_left = (context_window-len(sent.text)) // 2
            #     add_right = (context_window-len(sent.text)) - add_left

            #     j = i - 1
            #     while j >= 0 and add_left > 0:
            #         context_to_add = doc[j].text[-add_left:]
            #         tokens = context_to_add + tokens
            #         add_left -= len(context_to_add)
            #         sent_start += len(context_to_add)
            #         sent_end += len(context_to_add)
            #         j -= 1

            #     j = i + 1
            #     while j < len(doc) and add_right > 0:
            #         context_to_add = doc[j].text[:add_right]
            #         tokens = tokens + context_to_add
            #         add_right -= len(context_to_add)
            #         j += 1
            
            if context_window>0:
                original_sentences = doc.original_sentences
                nsent_to_add = context_window
                start_sent_id, end_sent_id = gold_rel.sent1_id, gold_rel.sent2_id
                
                while nsent_to_add > 0 and len(tokens)< max_seq_length-100:
                    left_sent, right_sent = [], []
                    if start_sent_id - 1 > 0:
                        start_sent_id -= 1
                        left_sent = original_sentences[start_sent_id]
                        sent_start += len(left_sent)
                        sent_end += len(left_sent)
                    if end_sent_id + 1 < len(original_sentences):
                        end_sent_id += 1
                        right_sent = original_sentences[end_sent_id]
                    
                    tokens = left_sent + tokens + right_sent
                    #left_sent, right_sent = original_sentences
                    nsent_to_add -= 1
                    assert sent.text[sub.span.start_sent]==tokens[sub.span.start_sent + sent_start]
                # debug: finish adding, and test
                
            
            


            # obj, sub, label, sent_start, sent_end, tokens, doc_id, lid
            if label != "":
                sample1 = set_sample(sub, obj, label, sent_start, sent_end, tokens, doc._doc_key, gold_rel.id, sent.sentence_ix)
                sent_samples.append(sample1)
                # DEBUG: fix this, only augment in training data, only augment 'AFTER' ~9.5%
                if task=="i2b2":
                    if ('train' in entity_data) and label in (['BEFORE']):#['BEFORE', 'AFTER']:
                        flipped_label = 'AFTER' if label=='BEFORE' else 'BEFORE'
                        sample2 = set_sample(obj, sub, flipped_label, sent_start, sent_end, tokens, doc._doc_key, gold_rel.id+"_flipped", sent.sentence_ix)
                        sent_samples.append(sample2)
                if task=="tbd":
                    if ('train' in entity_data) and label in (['SIMULTANEOUS']):#['BEFORE', 'AFTER']:
                        flipped_label = 'SIMULTANEOUS'
                        sample2 = set_sample(obj, sub, flipped_label, sent_start, sent_end, tokens, doc._doc_key, gold_rel.id+"_flipped", sent.sentence_ix)
                        sent_samples.append(sample2)
            
            
            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    print(f"nrel before flipped: {nrel}")
    logger.info('#nrel after flipped samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    return data, samples, nrel