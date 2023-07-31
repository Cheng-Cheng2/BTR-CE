import spacy, pdb, json, os, glob, argparse
#import pandas as pd
import pickle
import re 
#from timegraph.temporal_evaluation_adapted import event_is_e
def get_composite_marker (ner, dep):
    ner_dep = [] 
    for i, (n_pair, p_pair) in enumerate(zip(ner, dep)): 
        marker_id = 2
        n_pair_1, n_pair_2 = n_pair
        p_pair_1, p_pair_2 = p_pair
        ner_dep_pair1 = n_pair_1[:marker_id] + [f"{n_pair_1[marker_id]}:{p_pair_1[marker_id]}"]
        ner_dep_pair2 = n_pair_2[:marker_id] + [f"{n_pair_2[marker_id]}:{p_pair_2[marker_id]}"]
        ner_dep.append([ner_dep_pair1, ner_dep_pair2])
    
    assert len(ner_dep)==len(ner)    
    return ner_dep    
def custom_tokenizer(nlp):
    from spacy.tokenizer import Tokenizer
    from spacy.util import compile_infix_regex
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

def event_is_e(r_str, eid):
    eid=eid.lower()
    search_result = re.search(r_str, eid)
    if search_result is None:
        return False
    if re.search(r_str, eid).group(0)==eid:
        return True
    return False

def get_event_doc_span(st, ed, doc):    
    """_summary_

    Args:
        st (int): char level start id 
        ed (int): char level end id 
        doc (spacy.doc): 

    Returns:
        doc level span
        
    Note: a lot of weird cases for event=S1 (discharge) that needs shifting
    """
    # weird case in 333.xml, S1
    # wrong_sts=
    # wrong_eds=
    # wrong_dates= 
    if st in [42, 43] and ed in [49, 51] and doc.text[st:ed] in ['-15-93 ', '-18-94 ', '7/16/95 ', '2-10-93 ', '-12-93 ', '8-22-93 ']:
        st -= 1
        ed -= 1
        
        
    if st in [17, 45] and ed in [27, 55] and doc.text[st:ed] in ['011-02-08 ','011-02-14 ', '014-03-31 ', '014-04-01 ']:
        st -= 1
        ed -= 1
    if st in [17] and ed in [27] and doc.text[st:ed] in ['20041031 D', '20010614 D', '20041012 D']:
        #st -= 1
        ed -= 2
        
    if st in [45] and ed in [55] and doc.text[st:ed] in ['041103 ASS', '010616 ASS', '041015 ASS']:
        st-=2
        ed-=4
    span = doc.char_span(st, ed)
    doc_span = [span.start, span.end]
    assert doc[doc_span[0]:doc_span[1]].text== doc.text[st:ed]
    return doc_span




# NOTE: adjusted doc_span end to match PEER
def get_sentence_entities(l, doc):
    """
    return 
        - sentences_tokens : [[t_1, t_2,...], ..]
        - entities_dict: dict['eid'] = [[span.s, span.e, type, eType, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0, 0],...}
        - sent_start_map: dict[sent_id] = sent_start (doc_level)
    """
    # set doc_span for each entity
    for e_k, e_v in l['entities'].items():
        doc_span = get_event_doc_span(e_v['span'][0], e_v['span'][1], doc)
        doc_span[-1] = doc_span[-1] - 1 # NOTE: adjusted doc_span end to match PEER
        e_v['doc_span'] = doc_span   
    entities = sorted(l['entities'].items(), key=lambda x: x[1]['doc_span'][0])

    sentence_tokens = []
    #sent_span = []
    sent_start_map = {}
    entities_dict = {}
    e_idx = 0
    #entities_retokenize_dict = {}
    for sent_id, sent in enumerate(doc.sents):
        sentence_tokens.append([token.text for token in sent])
        #print( [sent.start_char, sent.end_char], [sent.start, sent.end], sent)
        #sent_span.append([sent.start, sent.end])
        sent_start_map[sent_id] = sent.start
        while e_idx < len(entities) and entities[e_idx][1]['doc_span'][0] < sent.end: 
            e_k, e_v = entities[e_idx]
            entities_dict[e_k] = e_v['doc_span']+[e_v['type'], e_v['eType']] + [sent_id] + [sent.start] # DEBUG: this sent.start are used to create sent_adjusted 
            
            e_idx += 1

    assert len(entities) == e_idx == len(entities_dict)
    return sentence_tokens,  entities_dict, sent_start_map
def merge_spans_for_pos(entities_dict, doc, debug_entities):
    """_summary_
    
    - entities_dict: dict['eid'] = [[span.s, span.e, type, eType, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0, 0],...}
            
    - return a dict with {'EO': ['pos', 'dep']}
    """
    entities_pos_dep = {}
    move_back_steps = 0
    for e_id, e_v in entities_dict.items():
        old_doc_start, old_doc_end, _, _, _, _ = e_v
        tokens, poss, deps = [], [], []
        doc_start, doc_end = old_doc_start - move_back_steps, old_doc_end - move_back_steps
        if doc_end-doc_start > 0:
            old_text = debug_entities[e_id]['text']
            old_doc_span = debug_entities[e_id]['doc_span']
            with doc.retokenize() as retokenizer:
                retokenizer.merge(doc[doc_start:doc_end+1])
                #token = doc[token_id]
                # pos =  token.pos_
                # dep = token.dep_
            new_token = doc[doc_start]
            #assert new_token.text == old_text
            pos =  new_token.pos_
            dep = new_token.dep_
            move_back_steps += doc_end - doc_start
            #print(new_token)
        else:
            token = doc[doc_start]
            pos =  token.pos_
            dep = token.dep_
                    
        entities_pos_dep[e_id] = [pos, dep]    
    return entities_pos_dep
def get_relations(l, all_entities, sent_start_map):
    
    """_summary_
    args:
        - l: dict_keys(['id', 'raw_text', 'entities', 'relations'])
        - all entities: {'E0': [0, 1, 'OCCURRENCE', 'EVENT', 0, 0], ...} 
    return: 
        - relations:
        [('TL201', 'E2', 'E3', 58, 58):[861, 862, 853, 857, 'ee', 'OVERLAP'],...,('SECTIME163', 11, 0):[154, 156, 3, 4, 'es', 'BEFORE', 'SECTIME163'],...)]
        i.e. [(lid, sent_id1, sent_id2):(s1_s, s1_e, s2_s, s2_e, 'l_type', relation, lid)]
        
    # BERT was pretrained using the format [CLS] sen A [SEP] sen B [SEP]
    """
    relations = {}
    for link_id, v in l['relations'].items():
        e1_id, e2_id = v['fromID'], v['toID']
        e1, e2 = all_entities[e1_id], all_entities[e2_id]
        e_sent_id = 4
        #relations[(link_id, e1_id, e2_id, e1[e_sent_id], e2[e_sent_id])] = e1[:2] + e2[:2] + [v['tlType'], v['type']] #+ [link_id]
        relations[(link_id, e1_id, e2_id, e1[e_sent_id], e2[e_sent_id])] = e1[:2] + e2[:2] + [v['tlType'], v['type'], v['secType']] 
   # print(f"org nrel:{len(l['relations'])}, formated nrel:{len(relations)}")
    
    assert len(l['relations'])==len(relations)
    
    
    return relations

def get_link_type_counts(relations, relations_values):
    rel_types = ['SEC', 'ET_OTHER', 'ET', 'TT', 'EE']
    count_link_by_type = {t:0 for t in rel_types}
    for k, v in zip(relations, relations_values):
        lid, e1, e2, sent_id1, sent_id2 = k
        #s1_s, s1_e, s2_s, s2_e, l_type, relation, lid = v
        if "sec" in lid.lower():
            lType = 'SEC'
            count_link_by_type[lType] += 1
            count_link_by_type['ET'] += 1
        else:
            r_e_type = r"(e)(\d+)"
            if (event_is_e(r_e_type, e1) and not event_is_e(r_e_type, e2)) or (not event_is_e(r_e_type, e1) and event_is_e(r_e_type, e2)):
                lType = 'ET_OTHER'
                count_link_by_type[lType] += 1
                count_link_by_type['ET'] += 1
            elif not event_is_e(r_e_type, e1) and not event_is_e(r_e_type, e2):
                lType = 'TT'
                count_link_by_type[lType] += 1
            else: # ee      
                lType = 'EE'
                count_link_by_type[lType] += 1
            
    return count_link_by_type

def format_entities(entities, L):
    """_summary_

    Args:
        entities (_type_): dict['eid'] = [[span.s, span.e, type, eType, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0, 0],...}
        L (int): num of sentences

    Returns: they are all lists of shape L, each line of the list correspond to a sentence by id
        ner (list(list(token)))
        ner_plus_time (list(list(token)))
        etype (list(list(token)))
    """
    ner = [[] for i in range(L)]
    ner_plus_time = [[] for i in range(L)]
    etype = [[] for i in range(L)]

    type_id, etype_id = 2, 3
    for k, v in entities.items():
        # NOTE: removed v[1] -= 1, because span end is adjusted before
        etype[v[-1]].append(v[:type_id] + [v[etype_id]])
        
        # for ner, 
        if v[etype_id] == 'TIMEX3':
            ner[v[-1]].append(v[:type_id] + ['TIMEX'])
        else: # EVENT or Sectime: ADMISSION/DISCHARGE
            ner[v[-1]].append(v[:type_id] + [v[type_id]])
        ner_plus_time[v[-1]].append(v[:type_id] + [v[type_id]])
    return ner, ner_plus_time, etype

def format_one_entity_and_append(entities, e_id, num_lk, ner, ner_plus_time, etype, secType="", isSecTime=False):
    """_summary_

    Args:
        entities (dict): dict['eid'] = [[span.s, span.e, type, etype, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0, 0],...}
        e_id

    Returns:  with current event added, each has doc_span
        ner: [0, 0, 'OCCURRENCE', TIMEX/ner]
        ner_plus_time:[0, 0, 'OCCURRENCE', ner]
        etype:[0, 0, 'OCCURRENCE', EVENT/TIMEX3/SECTIME]
    """
    # ner = [[] for i in range(L)]
    # ner_plus_time = [[] for i in range(L)]
    # etype = [[] for i in range(L)]
    event = entities[e_id]
    type_id, etype_id = 2, 3
    etype[num_lk].append(event[:type_id] + [event[etype_id]])
    

    # Note: for NER    
    if event[etype_id] == 'TIMEX3':
        ner[num_lk].append(event[:type_id] + ['TIMEX3'])
    else: # EVENT or Sectime: ADMISSION/DISCHARGE
        if isSecTime:
            ner[num_lk].append(event[:type_id] + [secType])
        else:
            ner[num_lk].append(event[:type_id] + [event[type_id]])
    
    # Note: for NER plus time
    # DEBUG: event[etype_id] not in ['TIMEX3', 'EVENT'] never happends?
    if isSecTime: # 
        ner_plus_time[num_lk].append(event[:type_id] + [secType]) # NOTE: secType is ADMISSION/DISCHARGE
    else: # Not events related to sectime
        ner_plus_time[num_lk].append(event[:type_id] + [event[type_id]]) # ? I want to change them to admission or discharge
        
    
    return ner, ner_plus_time, etype
def format_one_entity_pos_dep(entities, e_id, num_lk, pos, dep, pos_dep_dict, bert):
    """_summary_

    Args:
        entities (dict): dict['eid'] = [[span.s, span.e, type, etype, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0, 0],...}
        e_id

    Returns:  with current event added, each has doc_span
        ner: [0, 0, 'OCCURRENCE', TIMEX/ner]
        ner_plus_time:[0, 0, 'OCCURRENCE', ner]
        etype:[0, 0, 'OCCURRENCE', EVENT/TIMEX3/SECTIME]
    """

    event = entities[e_id]
    type_id, etype_id = 2, 3
    # etype[num_lk].append(event[:type_id] + [event[etype_id]])
    
    pos[num_lk].append(event[:type_id] + [pos_dep_dict[e_id][0]])
    dep[num_lk].append(event[:type_id] + [pos_dep_dict[e_id][1]])
    bert[num_lk].append(event[:type_id] + [""])


    return pos, dep, bert
    """_summary_
    - args:
        - relations:
        [('TL201', 'E2', 'E3', 58, 58):[861, 862, 853, 857, 'ee', 'OVERLAP', 'TL201'],...,('SECTIME163', 11, 0):[154, 156, 3, 4, 'es', 'BEFORE', 'SECTIME163'],...)]
        i.e. [(lid, sent_id1, sent_id2):(s1_s, s1_e, s2_s, s2_e, 'l_type', relation, lid)]
    
    - return:
        - formulated_rels (L x S x rel_s): [[[861, 862, 853, 857,'OVERLAP'], ...]]
    
    """

def formulate_relations_entities(sentences, relations, sent_start_map, entities, L, S=1, entities_pos_dep_dict=None):
    """_summary_

    Args:
        sentences (list(list)): list of tokens
        relations (_type_): [('TL201', 'E2', 'E3', 58, 58):[861, 862, 853, 857, 'ee', 'OVERLAP', 'TL201'],...,('SECTIME163', 11, 0):[154, 156, 3, 4, 'es', 'BEFORE', 'SECTIME163'],...)]
            i.e. [(lid, sent_id1, sent_id2):(s1_s, s1_e, s2_s, s2_e, 'l_type', relation, lid)]
        sent_start_map (map(sent_id):sent_start): _description_
        entities (_type_): dict['eid'] = [[span.s, span.e, type, eType, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0, 0],...}
        L (int): _description_
        S (int, optional): Defaults to 1.

    Returns: (all same len())
       formulated_rel: [[25, 25, 3, 3, 'BEFORE', 'SECTIME73', 188, 188, 3, 3],...],
            [e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, sent1_start, sent2_start, secType]
       tokens: [[t00, t01,...], ...]
       # NOTE: ner and below has doc_span, but rel has both sent and doc_span
       ner: [[[188, 188, 'OCCURRENCE'], [3, 3, 'TIMEX3']],...]
       ner_plus_time: [[[188, 188, 'OCCURRENCE'], [3, 3, 'DATE']], ...]
       etype: [[188, 188, 'EVENT'], [3, 3, 'TIMEX3']], ...]
    """
    FOR_BERT = True
    RELATION_TRUNCATE = True
    STEPS_INWARD = 2
    formulated_rels = []#[[] for i in range(L)]
    tokens = []
    ner = [[] for i in range(len(relations))]
    ner_plus_time = [[] for i in range(len(relations))]
    etype = [[] for i in range(len(relations))]
    pos = [[] for i in range(len(relations))]
    dep = [[] for i in range(len(relations))]
    bert = [[] for i in range(len(relations))]
    def update_doc_id_to_sent_id(doc_s, doc_e, sent_start):
        return doc_s - sent_start, doc_e - sent_start

    #num_l = 0
    for num_l, ((link_id, e1_id, e2_id, sent1_id, sent2_id), rel_v) in enumerate(relations.items()):
        #for i in range(L):
        e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, _, rel_val, secType = rel_v
        sent_tk1, sent_tk2 = sentences[sent1_id], sentences[sent2_id]
        curr_tokens = []
        
        e1_sent_s, e1_sent_e = update_doc_id_to_sent_id(e1_doc_s, e1_doc_e, sent_start_map[sent1_id])
        e2_sent_s, e2_sent_e = update_doc_id_to_sent_id(e2_doc_s, e2_doc_e, sent_start_map[sent2_id])    
        
        # if "sectime" in link_id.lower(): # concatenate two sentences            
        #     if sent1_id > sent2_id: 
        #         e1_sent_s, e1_sent_e = e1_sent_s+len(sentences[sent2_id]), e1_sent_e+len(sentences[sent2_id])
        #         curr_tokens = sent_tk2 + sent_tk1
        #     elif sent1_id < sent2_id:
        #         e2_sent_s, e2_sent_e = e2_sent_s+len(sentences[sent1_id]), e2_sent_e+len(sentences[sent1_id])
        #         curr_tokens = sent_tk1 + sent_tk2
        #     else:
        #         curr_tokens = sent_tk1
 
        #else: # concatenate everything within
        # NOTE: treat sectime and all other links the same
        # if not RELATION_TRUNCATE: # concatenate everything within
        #     new_sent1_id, new_sent2_id = sent1_id, sent2_id
        #     small_e_s, small_e_e = e1_sent_s, e1_sent_e
        #     large_e_s, large_e_e = e2_sent_s, e2_sent_e
        #     if sent1_id > sent2_id:
        #         new_sent1_id, new_sent2_id = sent2_id, sent1_id
        #         small_e_s, small_e_e = e2_sent_s, e2_sent_e
        #         large_e_s, large_e_e = e1_sent_s, e1_sent_e  
            
        #     for curr_sent_id in range(new_sent1_id, new_sent2_id):
        #         large_e_s += len(sentences[curr_sent_id])
        #         large_e_e += len(sentences[curr_sent_id])
                
        #         curr_tokens += sentences[curr_sent_id]
        #     curr_tokens += sentences[new_sent2_id]
        #     # final return
        #     if sent1_id > sent2_id:
        #         e1_sent_s, e1_sent_e = large_e_s, large_e_e
        #         e2_sent_s, e2_sent_e = small_e_s, small_e_e
        #     else:
        #         e1_sent_s, e1_sent_e = small_e_s, small_e_e
        #         e2_sent_s, e2_sent_e = large_e_s, large_e_e
        if FOR_BERT:
            new_sent1_id, new_sent2_id = sent1_id, sent2_id
            small_e_s, small_e_e = e1_sent_s, e1_sent_e
            large_e_s, large_e_e = e2_sent_s, e2_sent_e
            if sent1_id > sent2_id:
                new_sent1_id, new_sent2_id = sent2_id, sent1_id
                small_e_s, small_e_e = e2_sent_s, e2_sent_e
                large_e_s, large_e_e = e1_sent_s, e1_sent_e  
            
            # potential_ids = set([i for i in range(new_sent1_id, new_sent1_id+STEPS_INWARD+1)] + [i for i in range(new_sent2_id-STEPS_INWARD, new_sent2_id)])
            # assert len(potential_ids) <= 1 + 2*STEPS_INWARD 
            # for curr_sent_id in range(new_sent1_id, new_sent2_id):
            #     # NOTE: added the constraint
            #     if curr_sent_id in potential_ids:
            #         large_e_s += len(sentences[curr_sent_id])
            #         large_e_e += len(sentences[curr_sent_id])
                    
            #         curr_tokens += sentences[curr_sent_id]
                
            curr_tokens += sentences[new_sent2_id]
            
            # final return
            if sent1_id > sent2_id:
                e1_sent_s, e1_sent_e = large_e_s, large_e_e
                e2_sent_s, e2_sent_e = small_e_s, small_e_e
            else:
                e1_sent_s, e1_sent_e = small_e_s, small_e_e
                e2_sent_s, e2_sent_e = large_e_s, large_e_e
        
        else:
            new_sent1_id, new_sent2_id = sent1_id, sent2_id
            small_e_s, small_e_e = e1_sent_s, e1_sent_e
            large_e_s, large_e_e = e2_sent_s, e2_sent_e
            if sent1_id > sent2_id:
                new_sent1_id, new_sent2_id = sent2_id, sent1_id
                small_e_s, small_e_e = e2_sent_s, e2_sent_e
                large_e_s, large_e_e = e1_sent_s, e1_sent_e  
            
            potential_ids = set([i for i in range(new_sent1_id, new_sent1_id+STEPS_INWARD+1)] + [i for i in range(new_sent2_id-STEPS_INWARD, new_sent2_id)])
            assert len(potential_ids) <= 1 + 2*STEPS_INWARD 
            for curr_sent_id in range(new_sent1_id, new_sent2_id):
                # NOTE: added the constraint
                if curr_sent_id in potential_ids:
                    large_e_s += len(sentences[curr_sent_id])
                    large_e_e += len(sentences[curr_sent_id])
                    
                    curr_tokens += sentences[curr_sent_id]
                
            curr_tokens += sentences[new_sent2_id]
            # final return
            if sent1_id > sent2_id:
                e1_sent_s, e1_sent_e = large_e_s, large_e_e
                e2_sent_s, e2_sent_e = small_e_s, small_e_e
            else:
                e1_sent_s, e1_sent_e = small_e_s, small_e_e
                e2_sent_s, e2_sent_e = large_e_s, large_e_e
        
       
       
       
       
       
        final_sent_1_id, final_sent_2_id = sent1_id, sent2_id
        if sent1_id > sent2_id:
            final_sent_1_id, final_sent_2_id = sent2_id, sent1_id
        
        
        
        
        this_rel = [e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, final_sent_1_id, final_sent_2_id]
        
        formulated_rels.append(this_rel)
        tokens.append(curr_tokens)
        
        # formulate events
        # NOTE: For seclinks, the orders are always (E, T) 
        ner, ner_plus_time, etype = format_one_entity_and_append(entities, e1_id, num_l, ner, ner_plus_time, etype, secType, isSecTime=False)
        isSecTime = True if "sectime" in link_id.lower() else False
        ner, ner_plus_time, etype = format_one_entity_and_append(entities, e2_id, num_l, ner, ner_plus_time, etype, secType, isSecTime=isSecTime)
        
        #num_l += 1
        # Note: added pos, dep
        pos, dep, bert = format_one_entity_pos_dep(entities, e1_id, num_l, pos, dep, entities_pos_dep_dict, bert)
        pos, dep, bert = format_one_entity_pos_dep(entities, e2_id, num_l, pos, dep, entities_pos_dep_dict, bert)
        
    # add ner_pos, ner_dep, ner_pos_dep, ner_time_pos_dep

    
    ner_dep = get_composite_marker(ner, dep) 
    ner_pos = get_composite_marker(ner, pos)
    ner_dep_pos = get_composite_marker(ner_dep, pos)
    ner_time_dep = get_composite_marker(ner_plus_time, dep)  
    ner_time_dep_pos = get_composite_marker(ner_time_dep, pos)     
    ner_time_pos = get_composite_marker(ner_plus_time, pos)   
    
    assert len(formulated_rels)==len(tokens)==len(ner)==len(ner_plus_time)==len(etype)
    return tokens, formulated_rels, ner, ner_plus_time, etype, pos, dep, ner_dep, ner_pos, ner_dep_pos, ner_time_dep, ner_time_dep_pos, ner_time_pos, bert
    


""""
Input: 
    - entities_dict: dict['eid'] = [[span.s, span.e, type, eType, sent_id, sent_start]]
        - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0],...}
        - type: NER for event, DURATION/FREQ/etc for timex, ADMISSION/DISCHARGE for sectime
        - etype: EVENT/TIMEX/SECTIME
        
        
Return:
    - ner: using type in events and sectime, but 'time' for timex
    - ner_plus_time: using type in events and sectime, but also type in timex
    - etype: using etype: EVENT/TIMEX/SECTIME
"""

def format_entities(entities, L):
    ner = [[] for i in range(L)]
    ner_plus_time = [[] for i in range(L)]
    etype = [[] for i in range(L)]

    type_id, etype_id, sent_id = 2, 3, 4
    for k, v in entities.items():
        # NOTE: removed v[1] -= 1, because span end is adjusted before
        etype[v[sent_id]].append(v[:type_id] + [v[etype_id]])
        
        # for ner, 
        if v[etype_id] == 'TIMEX3':
            ner[v[sent_id]].append(v[:type_id] + ['TIMEX'])
        else: # EVENT or Sectime: ADMISSION/DISCHARGE
            ner[v[sent_id]].append(v[:type_id] + [v[type_id]])
        ner_plus_time[v[sent_id]].append(v[:type_id] + [v[type_id]])
    return ner, ner_plus_time, etype


# this was for the other task in CHARM
def format_event_timex(entities, L):
    format_entities_e = [[] for i in range(L)]
    format_entities_t = [[] for i in range(L)]
    for e, v in entities.items():
        v[1] -= 1
        if 'E' in e:
            format_entities_e[v[-1]].append(v[:-2])
        else:
            format_entities_t[v[-1]].append(v[:-2])    
    return format_entities_e, format_entities_t


def get_args(parser):
    parser.add_argument('--type_file', type=str, default="test.json", 
                        help="path to the file to load")
    # parser.add_argument('--output_dir', type=str, default='entity_output', 
    #                     help="output directory of the entity model")
    args = parser.parse_args()
    return args
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    
    
    
    #TRAINNING_DATA_DIR = "./corpus/i2b2/2012-07-15.original-annotation.release/"
    #TEST_DATA_DIR = "./corpus/i2b2/ground_truth/merged_xml/"
    LOAD_DIR = "./corpus/i2b2/"
    #if ENTITY:   
    #SAVE_DIR = "./corpus/i2b2/relation"
    SAVE_DIR = "/data/chengc7/MarkerTRel/i2b2_data/relations_bert"
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    type_file = args.type_file


    json_file = os.path.join(LOAD_DIR, type_file)
    dev_split = set(['86', '156', '203', '256', '318', '417', '473', '637', '681'])
    
    
    lines = []
    for l in open(json_file):
        l = json.loads(l)
        lines.append(l)
        

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")

    #debug: stop splitting both hyphenated numbers and words into separate tokens?

        
    nlp.tokenizer = custom_tokenizer(nlp)

    final_output = []
    total_relations = 0
    ORIGINAL_NREL = 0
    ENTITY = False
    all_links = []
    all_links_values  = []
    save_doc_link_ee_dict = {}
    for l in lines:
        s = l['raw_text']
        ORIGINAL_NREL += len(l['relations'])
        doc = nlp(s)
        sentences, entities, sent_start_map = get_sentence_entities(l, doc)
        relations = get_relations(l, entities, sent_start_map)
        save_doc_link_ee_dict[l['id'][0]] = {}
        #count = 0
        how_many_duplicate = 0
        for lid, e1, e2, _, _ in relations.keys():
        #('TL201', 'E2', 'E3', 58, 58)
            # if lid not in save_doc_link_ee_dict[l['id'][0]]:
            save_doc_link_ee_dict[l['id'][0]][lid] = (e1, e2)
            # else:
            #     how_many_duplicate += 1
            #     save_doc_link_ee_dict[l['id'][0]][f"lid_{how_many_duplicate}"] = (e1, e2)
        all_links += relations.keys()
        all_links_values += relations.values()
        L = len(sentences)
        entities_pos_dep = merge_spans_for_pos(entities, doc, l['entities'])
        if not ENTITY:
            #ners, ner_plus_times, etypes = format_entities(entities, L)
            #formulate_relations(sentences, relations, sent_start_map, entities, L, S=1)
            tokens, formulated_rels, ner, ner_plus_time, etype, pos, dep, ner_dep, ner_pos, ner_dep_pos, ner_time_dep, ner_time_dep_pos, ner_time_pos, bert = formulate_relations_entities(sentences, relations, sent_start_map, entities, L, entities_pos_dep_dict=entities_pos_dep)
            final_output.append({'doc_key':l['id'][0], 'sentences':tokens, 'relations':formulated_rels ,'ner':ner, 'ner_plus_time':ner_plus_time, 'etype':etype, 'pos':pos, 'dep':dep, 'ner_dep':ner_dep, 'ner_pos':ner_pos,'ner_dep_pos': ner_dep_pos, 'ner_time_dep':ner_time_dep,'ner_time_dep_pos':ner_time_dep_pos, 'ner_time_pos':ner_time_pos, 'original_sentences':sentences, "bert":bert})
        
        else:

              # this might not be working now...
            new_e, new_t = format_event_timex(entities, len(sentences)) 
            final_output.append({'doc_key':l['id'][0], 'sentences':sentences, 'ner':new_e, 'entities_t': new_t})
            
        
        print("doc_id:", l['id'], " n_entites:", len(entities), " n_relations:", len(relations))
        total_relations += len(relations)
        # print("ORIGNAL_NREL", ORIGINAL_NREL)
        # assert ORIGINAL_NREL == total_relations
        #print("doc_id:", l['id'])

    # save   
    import pandas as pd
    if args.type_file == "test.json":
        # NOTE: save count for link type
        link_count_by_type_dict = get_link_type_counts(all_links, all_links_values)
        df = pd.DataFrame.from_dict(link_count_by_type_dict, orient="index")  
        df.to_csv(os.path.join(SAVE_DIR, "test_link_count_by_type.csv"))
        with open(os.path.join(SAVE_DIR, "doc_lid_ee_map.json"), "w") as f:
            json.dump(save_doc_link_ee_dict, f)
        # NOTE: save count for marker type
        
    #print(f"{type_file} {len(relations)}: {total_relations}")
    print(f"{type_file} {len(relations)}: {total_relations}")
    if not ENTITY:
        if 'test' in type_file:
            save_file = os.path.join(SAVE_DIR, type_file)
            json_str = "\n".join([json.dumps(l) for l in final_output])
            with open(save_file, 'w') as f:
                f.write(json_str)
                        
            print(f"{type_file}: final n: {len(final_output)}")
        else:
            two_files = ['train.json', 'dev.json']
            #save_files = [os.path.join(SAVE_DIR, i) for i in two_files]
            final_out = {'train.json':[], 'dev.json':[]}
            final_nrel = {'train.json':0, 'dev.json':0}
            for i, d in enumerate(final_output):
                #pdb.set_trace()
                if d['doc_key'] not in dev_split:
                    final_out['train.json'].append(d)
                    final_nrel['train.json'] += len(d['relations'])
                else:
                    final_out['dev.json'].append(d)                
                    final_nrel['dev.json'] += len(d['relations'])
                
                # save them
            for type_file in two_files:
                save_file = os.path.join(SAVE_DIR, type_file)
                json_str = "\n".join([json.dumps(l) for l in final_out[type_file]])
                with open(save_file, 'w') as f:
                    f.write(json_str)
                    
            print("nrel", final_nrel)    
            print(f"train.json: final n: {len(final_out['train.json'])}")
            print(f"dev.json: final n: {len(final_out['dev.json'])}")

