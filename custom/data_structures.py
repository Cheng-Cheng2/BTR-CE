"""
This code is based on DYGIE++'s codebase
"""
import json
import copy
import os
from collections import Counter
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

def fields_to_batches(d, keys_to_ignore=[]):
    keys = [key for key in d.keys() if key not in keys_to_ignore]
    lengths = [len(d[k]) for k in keys] # I think this is length of each value in dic
    #print(set(lengths))
    assert len(set(lengths)) == 1 # make sure len of each val list in doc is the same
    length = lengths[0]
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res

def get_sentence_of_span(span, sentence_starts, doc_tokens):
    """
    Return the index of the sentence that the span is part of.
    """
    # Inclusive sentence ends
    sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_between = [span[0] >= start and span[1] <= end
                  for start, end in zip(sentence_starts, sentence_ends)]
    assert sum(in_between) == 1
    the_sentence = in_between.index(True)
    return the_sentence


class Dataset:
    def __init__(self, json_file, pred_file=None, doc_range=None, entity_type='ner', context_window=0): # added entity_type here
        # NOTE: when being called, pred file and doc_range seems to always be None
        self.js = self._read(json_file)
        self.entity_type = entity_type
        # if doc_range is not None:
        #     self.js = self.js[doc_range[0]:doc_range[1]]
        self.documents = [
            Document(js, entity_type=self.entity_type, context_window=context_window) for js in self.js]
        # if context_window > 0:
        #     self.orignal_sentences 

    def update_from_js(self, js):
        self.js = js
        self.documents = [
            Document(js, entity_type=self.entity_type) for js in self.js]

    def _read(self, json_file, pred_file=None):
        gold_docs = [json.loads(line) for line in open(json_file)]
        #if pred_file is None: # NOTE: alwyas the case
        return gold_docs

        # pred_docs = [json.loads(line) for line in open(pred_file)]
        # merged_docs = []
        # for gold, pred in zip(gold_docs, pred_docs):
        #     assert gold["doc_key"] == pred["doc_key"]
        #     assert gold["sentences"] == pred["sentences"]
        #     merged = copy.deepcopy(gold)
        #     for k, v in pred.items():
        #         if "predicted" in k:
        #             merged[k] = v
        #     merged_docs.append(merged)

        # return merged_docs

    def __getitem__(self, ix):
        return self.documents[ix]

    def __len__(self):
        return len(self.documents)


"""
each file:
{'doc_key':l['id'][0],
 'sentences':tokens,
 'relations':formulated_rels ,
 'ner':ner,
 'ner_plus_time':ner_plus_time,
 'etype':etype}


    relations: [[25, 25, 3, 3, 'BEFORE', 'SECTIME73', 188, 188, 3, 3],...],
        [e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e]
    sentences: [[t00, t01,...], ...]
    # NOTE: ner and below has doc_span, but rel has both sent and doc_span
    ner: [[[188, 188, 'OCCURRENCE'], [3, 3, 'TIMEX3']],...]
    ner_plus_time: [[[188, 188, 'OCCURRENCE'], [3, 3, 'DATE']], ...]
    etype: [[188, 188, 'EVENT'], [3, 3, 'TIMEX3']], ...]
"""
class Document:
    def __init__(self, js, entity_type="ner", context_window=0):
        self._doc_key = js["doc_key"]
        # REVIEW: why is "predicted_relations not ignored here?"
        entries = fields_to_batches(js, keys_to_ignore=["doc_key", "clusters", "predicted_clusters", "section_starts", "original_sentences"])

        # NOTE: entries is a list of dicts of each sentence, each entry is like: {'sentences':[w1, w2..],'ner':[], 'relations':[], 'predicted_ner':[], 'predicted_relations';[]}
        # NOTE: note the 'sentences' now is actually a list of tokens in one sentence
        
        # sentence_lengths = [len(entry["sentences"]) for entry in entries] # eg. 2, 3, 5
        # sentence_starts = np.cumsum(sentence_lengths) # eg. 2, 5, 20
        # sentence_starts = np.roll(sentence_starts, 1) # e.g.
        # sentence_starts[0] = 0 
        #self.sentence_starts = sentence_starts # NOTE: the sentence start in the whole document
        self.sentences = [Sentence(entry, sentence_ix, entity_type=entity_type, context_window=context_window)
                          for sentence_ix, entry in enumerate(entries)]#TODO : understand Sentence()
        if context_window > 0:
            self.original_sentences = [s for s in js['original_sentences']]
        # Note : clusteres are coreference annotations, not used in this method so could be ignored
        # if "clusters" in js:
        #     self.clusters = [Cluster(entry, i, self)
        #                      for i, entry in enumerate(js["clusters"])]
        # if "predicted_clusters" in js:
        #     self.predicted_clusters = [Cluster(entry, i, self)
        #                                for i, entry in enumerate(js["predicted_clusters"])]
        # Note close - ignored part
    def __repr__(self):
        return "\n".join([str(i) + ": " + " ".join(sent.text) for i, sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))


    def find_cluster(self, entity, predicted=True):
        """
        Search through erence clusters and return the one containing the query entity, if it's
        part of a cluster. If we don't find a match, return None.
        """
        clusters = self.predicted_clusters if predicted else self.clusters
        for clust in clusters:
            for entry in clust:
                if entry.span == entity.span:
                    return clust

        return None

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])

"""
each file:
{'doc_key':l['id'][0],
 'sentences':tokens,
 'relations':formulated_rels ,
 'ner':ner,
 'ner_plus_time':ner_plus_time,
 'etype':etype}


    relations: [[25, 25, 3, 3, 'BEFORE', 'SECTIME73', 188, 188, 3, 3],...],
        [e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e]
    sentences: [[t00, t01,...], ...]
    # NOTE: ner and below has doc_span, but rel has both sent and doc_span
    ner: [[[188, 188, 'OCCURRENCE'], [3, 3, 'TIMEX3']],...]
    ner_plus_time: [[[188, 188, 'OCCURRENCE'], [3, 3, 'DATE']], ...]
    etype: [[188, 188, 'EVENT'], [3, 3, 'TIMEX3']], ...]
    
    
    return:
    self.relations = [Relation] (1)
    self.ner = [NER, NER] (2)
"""
class Sentence:
    def __init__(self, entry, sentence_ix, entity_type, context_window=0):
        #self.sentence_start = sentence_start
        self.text = entry["sentences"]
        self.sentence_ix = sentence_ix # NOTE: NO USE CURRENTLY
        self.entity_type = entity_type
        # if context_window==0:
        #    e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e = entry["relations"]
        #else:
        
        e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, e1_sent_id, e2_sent_id = entry["relations"]
        # debug: change here to add sent1_id, sent2_id
        self.relations = [Relation(entry["relations"], self.text, context_window=context_window)]
        ner_sent_starts = [e1_sent_s, e2_sent_s]
        ner_sent_ends = [e1_sent_e, e2_sent_e]
        if self.entity_type in entry.keys():
            # (self, ner, text, sent_start, sent_end, flavor=None):
            self.ner = [NER(this_ner, self.text, sent_start=sent_s, sent_end=sent_e)
                        for this_ner, sent_s, sent_e in 
                            zip(entry[self.entity_type], ner_sent_starts, ner_sent_ends)] # this should give two
        if "predicted_relations" in entry.keys():
            pred_rel = entry['predicted_relations']
            if len(pred_rel) > 0:
                _, _, _, _, pred_rel_val = pred_rel[0]
                pred_rel_list = [e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, pred_rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, e1_sent_id, e2_sent_id]
                # debug: change here to add sent1_id, sent2_id copying from above
                self.predicted_relations=[Relation(pred_rel_list, self.text, context_window=context_window)]
       
    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)

    def get_flavor(self, argument):
        the_ner = [x for x in self.ner if x.span == argument.span]
        if len(the_ner) > 1:
            print("Weird")
        if the_ner:
            the_flavor = the_ner[0].flavor
        else:
            the_flavor = None
        return the_flavor

# NOTE: the span object has both doc idx (start_doc, end_doc) and sent idx (start_sent, end_sent)
# DEBUG: removed all need for sentence_start
class Span:
    def __init__(self, doc_start, doc_end, text, sent_start, sent_end):
        self.start_doc = doc_start
        self.end_doc = doc_end
        self.span_doc = (self.start_doc, self.end_doc)
        self.start_sent = sent_start #start - sentence_start
        self.end_sent = sent_end #end - sentence_start
        self.span_sent = (self.start_sent, self.end_sent)
        self.text = text[self.start_sent:self.end_sent + 1]

    def __repr__(self): 
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (self.span_doc == other.span_doc and
                self.span_sent == other.span_sent and
                self.text == other.text)

    def __hash__(self):
        tup = self.span_doc + self.span_sent + (" ".join(self.text),)
        #tup = self.span_sent + (" ".join(self.text),)
        return hash(tup)


class Token:
    def __init__(self, ix, text, sentence_start):
        self.ix_doc = ix
        self.ix_sent = ix - sentence_start
        self.text = text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))


class Trigger:
    def __init__(self, token, label):
        self.token = token
        self.label = label

    def __repr__(self):
        return self.token.__repr__()[:-1] + ", " + self.label + ")"


class Argument:
    def __init__(self, span, role, event_type):
        self.span = span
        self.role = role
        self.event_type = event_type

    def __repr__(self):
        return self.span.__repr__()[:-1] + ", " + self.event_type + ", " + self.role + ")"

    def __eq__(self, other):
        return (self.span == other.span and
                self.role == other.role and
                self.event_type == other.event_type)

    def __hash__(self):
        return self.span.__hash__() + hash((self.role, self.event_type))


class NER:
    def __init__(self, ner, text, sent_start, sent_end, flavor=None):
        self.span = Span(ner[0], ner[1], text, sent_start, sent_end)
        self.label = ner[2]
        self.flavor = flavor

    def __repr__(self):
        return self.span.__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.span == other.span and
                self.label == other.label and
                self.flavor == other.flavor)


class Relation:
    def __init__(self, relation, text, context_window=0):
        e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, e1_sent_id, e2_sent_id= relation
        # DEBUG: change here to add sent1_id, sent2_id
        # start1, end1 = relation[0], relation[1]
        # start2, end2 = relation[2], relation[3]
        # label = relation[4]
        # doc_start, doc_end, text, sent_start, sent_end
        span1 = Span(e1_doc_s, e1_doc_e, text, e1_sent_s, e1_sent_e)
        span2 = Span(e2_doc_s, e2_doc_e, text, e2_sent_s, e2_sent_e)
        self.pair = (span1, span2)
        self.label = rel_val
        self.id = link_id
        if context_window > 0:
            self.sent1_id = e1_sent_id
            self.sent2_id = e2_sent_id
    def __repr__(self):
        return self.pair[0].__repr__() + ", " + self.pair[1].__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.pair == other.pair) and (self.label == other.label)


class AtomicRelation:
    def __init__(self, ent0, ent1, label):
        self.ent0 = ent0
        self.ent1 = ent1
        self.label = label

    @classmethod
    def from_relation(cls, relation):
        ent0 = " ".join(relation.pair[0].text)
        ent1 = " ".join(relation.pair[1].text)
        label = relation.label
        return cls(ent0, ent1, label)

    def __repr__(self):
        return f"({self.ent0} | {self.ent1} | {self.label})"



class Event:
    def __init__(self, event, text, sentence_start):
        trig = event[0]
        args = event[1:]
        trigger_token = Token(trig[0], text, sentence_start)
        self.trigger = Trigger(trigger_token, trig[1])

        self.arguments = []
        for arg in args:
            span = Span(arg[0], arg[1], text, sentence_start)
            self.arguments.append(Argument(span, arg[2], self.trigger.label))

    def __repr__(self):
        res = "<"
        res += self.trigger.__repr__() + ":\n"
        for arg in self.arguments:
            res += 6 * " " + arg.__repr__() + ";\n"
        res = res[:-2] + ">"
        return res


class Events:
    def __init__(self, events_json, text, sentence_start):
        self.event_list = [Event(this_event, text, sentence_start) for this_event in events_json]
        self.triggers = set([event.trigger for event in self.event_list])
        self.arguments = set([arg for event in self.event_list for arg in event.arguments])

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, i):
       return self.event_list[i]

    def __repr__(self):
        return "\n\n".join([event.__repr__() for event in self.event_list])

    def span_matches(self, argument):
        return set([candidate for candidate in self.arguments
                    if candidate.span.span_sent == argument.span.span_sent])

    def event_type_matches(self, argument):
        return set([candidate for candidate in self.span_matches(argument)
                    if candidate.event_type == argument.event_type])

    def matches_except_event_type(self, argument):
        matched = [candidate for candidate in self.span_matches(argument)
                   if candidate.event_type != argument.event_type
                   and candidate.role == argument.role]
        return set(matched)

    def exact_match(self, argument):
        for candidate in self.arguments:
            if candidate == argument:
                return True
        return False


class Cluster:
    def __init__(self, cluster, cluster_id, document):
        members = []
        for entry in cluster:
            sentence_ix = get_sentence_of_span(entry, document.sentence_starts, document.n_tokens)
            sentence = document[sentence_ix]
            span = Span(entry[0], entry[1], sentence.text, sentence.sentence_start)
            ners = [x for x in sentence.ner if x.span == span]
            assert len(ners) <= 1
            ner = ners[0] if len(ners) == 1 else None
            to_append = ClusterMember(span, ner, sentence, cluster_id)
            members.append(to_append)

        self.members = members
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"{self.cluster_id}: " + self.members.__repr__()

    def __getitem__(self, ix):
        return self.members[ix]


class ClusterMember:
    def __init__(self, span, ner, sentence, cluster_id):
        self.span = span
        self.ner = ner
        self.sentence = sentence
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"<{self.sentence.sentence_ix}> " + self.span.__repr__()


####################

# Code to do evaluation of predictions for a loaded dataset.

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    # F1 score.
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)


def evaluate_sent(sent, counts):
    correct_ner = set()
    # Entities.
    # counts["ner_gold"] += len(sent.ner)
    # counts["ner_predicted"] += len(sent.predicted_ner)
    # for prediction in sent.predicted_ner:
    #     if any([prediction == actual for actual in sent.ner]):
    #         counts["ner_matched"] += 1
    #         correct_ner.add(prediction.span)

    # Relations.
    counts["relations_gold"] += len(sent.relations)
    counts["relations_predicted"] += len(sent.predicted_relations)
    for prediction in sent.predicted_relations:
        if any([prediction == actual for actual in sent.relations]):
            counts["relations_matched"] += 1
            if (prediction.pair[0] in correct_ner) and (prediction.pair[1] in correct_ner):
                counts["strict_relations_matched"] += 1

    # Return the updated counts.
    return counts

def evaluate_predictions(dataset):
    counts = Counter()

    for doc in dataset:
        for sent in doc:
            counts = evaluate_sent(sent, counts)

    scores_ner = compute_f1(
        counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"])
    scores_relations = compute_f1(
        counts["relations_predicted"], counts["relations_gold"], counts["relations_matched"])
    scores_strict_relations = compute_f1(
        counts["relations_predicted"], counts["relations_gold"], counts["strict_relations_matched"])

    return dict(ner=scores_ner, relation=scores_relations, strict_relation=scores_strict_relations)

def analyze_relation_coverage(dataset):
    
    def overlap(s1, s2):
        if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
            return True
        if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
            return True
        return False

    nrel_gold = 0
    nrel_pred_cover = 0
    nrel_top_cover = 0

    npair_pred = 0
    npair_top = 0

    nrel_overlap = 0

    for d in dataset:
        for s in d:
            pred = set([ner.span for ner in s.predicted_ner])
            top = set([ner.span for ner in s.top_spans])
            npair_pred += len(s.predicted_ner) * (len(s.predicted_ner) - 1)
            npair_top += len(s.top_spans) * (len(s.top_spans) - 1)
            for r in s.relations:
                nrel_gold += 1
                if (r.pair[0] in pred) and (r.pair[1] in pred):
                    nrel_pred_cover += 1
                if (r.pair[0] in top) and (r.pair[1] in top):
                    nrel_top_cover += 1
                
                if overlap(r.pair[0], r.pair[1]):
                    nrel_overlap += 1

    print('Coverage by predicted entities: %.3f (%d / %d), #candidates: %d'%(nrel_pred_cover/nrel_gold*100.0, nrel_pred_cover, nrel_gold, npair_pred))
    print('Coverage by top 0.4 spans: %.3f (%d / %d), #candidates: %d'%(nrel_top_cover/nrel_gold*100.0, nrel_top_cover, nrel_gold, npair_top))
    print('Overlap: %.3f (%d / %d)'%(nrel_overlap / nrel_gold * 100.0, nrel_overlap, nrel_gold))
