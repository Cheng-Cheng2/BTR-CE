
# TB-dense_data relatively small so I could consider running this first
import spacy
import json
import os
import sys
sys.path.append('.')
import pickle
import re
from formate_i2b2_for_bert import custom_tokenizer
def get_sentence_entities(doc, ents):
            """
            return 
                - sentences_tokens : [[t_1, t_2,...], ..]
                - entities_dict: dict['eid'] = [[span.s, span.e, pos, tense, polarity, sent_id, sent_start]]
                    - eg. {'E0': [0, 0, 'VERB', 'PRESPART', 'POS', 0, 0],...}    
                - sent_start_map: dict[sent_id] = sent_start (doc_level)
            """
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
                span = doc.char_span(st, ed)
                doc_span = [span.start, span.end]
                assert doc[doc_span[0]:doc_span[1]].text== doc.text[st:ed]
                return doc_span
            # set doc_span for each entity
            for e_k, e_v in ents.items():
                doc_span = get_event_doc_span(e_v['span'][0], e_v['span'][1]+1, doc) # debug: ed+1 is diff from i2b2
                doc_span[-1] = doc_span[-1] - 1 # NOTE: adjusted doc_span end to match PEER
                e_v['doc_span'] = doc_span   
            entities = sorted(ents.items(), key=lambda x: x[1]['doc_span'][0])

            sentence_tokens = []
            #sent_span = []
            sent_start_map = {}
            entities_dict = {}
            e_idx = 0
            
            for sent_id, sent in enumerate(doc.sents):
                sentence_tokens.append([token.text for token in sent])
                #print( [sent.start_char, sent.end_char], [sent.start, sent.end], sent)
                #sent_span.append([sent.start, sent.end])
                sent_start_map[sent_id] = sent.start
                while e_idx < len(entities) and entities[e_idx][1]['doc_span'][0] < sent.end: 
                    e_k, e_v = entities[e_idx]
                    entities_dict[e_k] = e_v['doc_span']+[e_v['pos'], e_v['tense'], e_v['polarity']] + [sent_id] + [sent.start] # DEBUG: diff from i2b2 no type eType 
                    e_idx += 1

            assert len(entities) == e_idx == len(entities_dict)
            return sentence_tokens,  entities_dict, sent_start_map
def get_relations(original_doc_rels, all_entities, sent_start_map):
        
    """_summary_
    args:
        - l: dict_keys(['id', 'raw_text', 'entities', 'relations'])
        - all entities: {'E0': [0, 1, 'OCCURRENCE', 'EVENT', 0, 0], ...} 
    return: 
        - relations:
        [('TL201', 'E2', 'E3', 58, 58):[861, 862, 853, 857, 'TLINK', 'INCLUDES'],...)]
        i.e. [(lid, sent_id1, sent_id2):(s1_s, s1_e, s2_s, s2_e, 'l_type', relation)]
        
    # BERT was pretrained using the format [CLS] sen A [SEP] sen B [SEP]
    """
    relations = {}
    for link_id, v in original_doc_rels.items():
        link_v = v['properties']
        assert len(link_v['source'])==len(link_v['target'])==1
        e1_id, e2_id = link_v['source'][0]['id'], link_v['target'][0]['id']
        e1, e2 = all_entities[e1_id], all_entities[e2_id]
        e_sent_id = 5
        assert len(link_v['type'])==1
        relations[(link_id, e1_id, e2_id, e1[e_sent_id], e2[e_sent_id])] = e1[:2] + e2[:2] + [v['type'], link_v['type'][0]] # debug: no secType
    
    assert len(original_doc_rels)==len(relations)   
    return relations        
def format_one_entity_and_append(entities, e_id, num_lk, pos, pos_tense, pos_tense_polarity, bert, secType="", isSecTime=False):
    """_summary_

    Args:
        entities (dict): dict['eid'] = [[span.s, span.e, type, etype, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'OCCURRENCE', 'EVENT', 0, 0],...}
        e_id
                - entities_dict: dict['eid'] = [[span.s, span.e, pos, tense, polarity, sent_id, sent_start]]
                    - eg. {'E0': [0, 0, 'VERB', 'PRESPART', 'POS', 0, 0],...}    
    Returns:  with current event added, each has doc_span
        ner: [0, 0, 'OCCURRENCE', TIMEX/ner]
        ner_plus_time:[0, 0, 'OCCURRENCE', ner]
        etype:[0, 0, 'OCCURRENCE', EVENT/TIMEX3/SECTIME]
    """
    event = entities[e_id]
    pos_id, pos_tense_id, pos_tense_polarity_id = 2, 3, 4
    pos_val, tense_val, polarity_val = event[pos_id], event[pos_tense_id], event[pos_tense_polarity_id]
    
    pos[num_lk].append(event[:pos_id] + [pos_val])
    pos_tense[num_lk].append(event[:pos_id] + [f"{pos_val}:{tense_val}"])
    pos_tense_polarity[num_lk].append(event[:pos_id] + [f"{pos_val}:{tense_val}:{polarity_val}"])

    # added
    bert[num_lk].append(event[:pos_id] + [""])
    # # Note: for NER    
    # if event[etype_id] == 'TIMEX3':
    #     ner[num_lk].append(event[:type_id] + ['TIMEX3'])
    # else: # EVENT or Sectime: ADMISSION/DISCHARGE
    #     if isSecTime:
    #         ner[num_lk].append(event[:type_id] + [secType])
    #     else:
    #         ner[num_lk].append(event[:type_id] + [event[type_id]])
    
    # # Note: for NER plus time
    # # DEBUG: event[etype_id] not in ['TIMEX3', 'EVENT'] never happends?
    # if isSecTime: # 
    #     ner_plus_time[num_lk].append(event[:type_id] + [secType]) # NOTE: secType is ADMISSION/DISCHARGE
    # else: # Not events related to sectime
    #     ner_plus_time[num_lk].append(event[:type_id] + [event[type_id]]) # ? I want to change them to admission or discharge
    
    return pos, pos_tense, pos_tense_polarity, bert  
def formulate_relations_entities(sentences, relations, sent_start_map, entities, L, S=1):
    """_summary_

    Args:
        sentences (list(list)): list of tokens
        relations (_type_): [('TL201', 'E2', 'E3', 58, 58):[861, 862, 853, 857, 'ee', 'OVERLAP', 'TL201'],...,('SECTIME163', 11, 0):[154, 156, 3, 4, 'es', 'BEFORE', 'SECTIME163'],...)]
            i.e. [(lid, sent_id1, sent_id2):(s1_s, s1_e, s2_s, s2_e, 'l_type', relation, lid)]
        sent_start_map (map(sent_id):sent_start): _description_
        - entities_dict: dict['eid'] = [[span.s, span.e, pos, tense, polarity, sent_id, sent_start]]
            - eg. {'E0': [0, 0, 'VERB', 'PRESPART', 'POS', 0, 0],...}    
        L (int): _description_
        S (int, optional): Defaults to 1.

    Returns: (all same len())
    formulated_rel: [[25, 25, 3, 3, 'BEFORE', 'SECTIME73', 188, 188, 3, 3],...],
            [e1_sent_s, e1_sent_e, e2_sent_s, e2_sent_e, rel_val, link_id, e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, sent1_start, sent2_start, secType]
    tokens: [[t00, t01,...], ...]
    # NOTE: ner and below has doc_span, but rel has both sent and doc_span
    pos: [[[188, 188, 'OCCURRENCE'], [3, 3, 'TIMEX3']],...]
    pos_tense: [[[188, 188, 'OCCURRENCE'], [3, 3, 'DATE']], ...]
    pos_tense_polarity: [[188, 188, 'EVENT'], [3, 3, 'TIMEX3']], ...]
    """
    formulated_rels = []#[[] for i in range(L)]
    tokens = []
    pos = [[] for i in range(len(relations))]
    pos_tense = [[] for i in range(len(relations))]
    pos_tense_polarity = [[] for i in range(len(relations))]
    bert = [[] for i in range(len(relations))]
    swap_label_order = {'OVERLAP':'OVERLAP', 'BEFORE':'AFTER', 'AFTER':'BEFORE'}
    def update_doc_id_to_sent_id(doc_s, doc_e, sent_start):
        return doc_s - sent_start, doc_e - sent_start

    #num_l = 0
    for num_l, ((link_id, e1_id, e2_id, sent1_id, sent2_id), rel_v) in enumerate(relations.items()):
        #for i in range(L):
        e1_doc_s, e1_doc_e, e2_doc_s, e2_doc_e, _, rel_val = rel_v
        sent_tk1, sent_tk2 = sentences[sent1_id], sentences[sent2_id]
        curr_tokens = []
        
        e1_sent_s, e1_sent_e = update_doc_id_to_sent_id(e1_doc_s, e1_doc_e, sent_start_map[sent1_id])
        e2_sent_s, e2_sent_e = update_doc_id_to_sent_id(e2_doc_s, e2_doc_e, sent_start_map[sent2_id])    
        
        if "sectime" in link_id.lower(): # concatenate two sentences            
            if sent1_id > sent2_id: 
                e1_sent_s, e1_sent_e = e1_sent_s+len(sentences[sent2_id]), e1_sent_e+len(sentences[sent2_id])
                curr_tokens = sent_tk2 + sent_tk1
            elif sent1_id < sent2_id:
                e2_sent_s, e2_sent_e = e2_sent_s+len(sentences[sent1_id]), e2_sent_e+len(sentences[sent1_id])
                curr_tokens = sent_tk1 + sent_tk2
            else:
                curr_tokens = sent_tk1

        else: # concatenate everything within
            new_sent1_id, new_sent2_id = sent1_id, sent2_id
            small_e_s, small_e_e = e1_sent_s, e1_sent_e
            large_e_s, large_e_e = e2_sent_s, e2_sent_e
            if sent1_id > sent2_id:
                new_sent1_id, new_sent2_id = sent2_id, sent1_id
                small_e_s, small_e_e = e2_sent_s, e2_sent_e
                large_e_s, large_e_e = e1_sent_s, e1_sent_e  
            
            for curr_sent_id in range(new_sent1_id, new_sent2_id):
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
        # NOTE: NO sectime here
            #DATA: - entities_dict: dict['eid'] = [[span.s, span.e, pos, tense, polarity, sent_id, sent_start]]
            # - eg. {'E0': [0, 0, 'VERB', 'PRESPART', 'POS', 0, 0],...}  
        
        pos, pos_tense, pos_tense_polarity, bert = format_one_entity_and_append(entities, e1_id, num_l, pos, pos_tense, pos_tense_polarity, bert, secType="", isSecTime=False)
        #isSecTime = True if "sectime" in link_id.lower() else False
        pos, pos_tense, pos_tense_polarity, bert = format_one_entity_and_append(entities, e2_id, num_l, pos, pos_tense, pos_tense_polarity, bert, secType="", isSecTime=False)

        #num_l += 1
    
    assert len(formulated_rels)==len(tokens)==len(pos)==len(pos_tense)==len(pos_tense_polarity)
    return tokens, formulated_rels, pos, pos_tense, pos_tense_polarity, bert

if __name__ == '__main__':            
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = custom_tokenizer(nlp) # NOTE: this was added
        
    temp_tb_file = "/data/chengc7/TEDataProcessing/tbd_output"
    js_lists = [json.loads(line) for line in open(temp_tb_file)]
    types = ["train", "dev", "test"]

    one_js = js_lists[0]
    print(one_js.keys())
    final_out = {'train':[], 'dev':[], 'test':[]}
    ndoc_dict, nrel_dict = {t:0 for t in types}, {t:0 for t in types}
    for i, json_doc in enumerate(js_lists):
        #for k, v in js_lists[t]:
            # ndoc[], nrel = 0, 0
            doc_id, raw_text, ents, rels, type = json_doc['id'], json_doc['raw_text'], json_doc["entities"], json_doc['relations'], json_doc['split'] 
            doc = nlp(raw_text)
            # entities: dict['eid'] = [[span.s, span.e, pos, tense, polarity, sent_id, sent_start]]
            #             - eg. {'E0': [0, 0, 'VERB', 'PRESPART', 'POS', 0, 0],...}
            sentences, entities, sent_start_map = get_sentence_entities(doc, ents)

            # data: relations:
            #         [('TL201', 'E2', 'E3', 58, 58):[861, 862, 853, 857, 'TLINK', 'INCLUDES'],...)]
            #         i.e. [(lid, sent_id1, sent_id2):(s1_s, s1_e, s2_s, s2_e, 'l_type', relation)]
            relations = get_relations(rels, entities, sent_start_map)

            tokens, formulated_rels, pos, pos_tense, pos_tense_polarity, bert = formulate_relations_entities(sentences, relations, sent_start_map, entities, len(sentences))
            final_out[type].append({'doc_key':doc_id, 'sentences':tokens, 'relations':formulated_rels ,'pos':pos, 'pos_tense':pos_tense, 'pos_tense_polarity':pos_tense_polarity, 'bert':bert, 'original_sentences':sentences})
            #print("test")
            ndoc_dict[type] += 1
            nrel_dict[type] += len(formulated_rels)
            
    test_docs = ["APW19980308.0201", "APW19980418.0210", "CNN19980213.2130.0155", "APW19980227.0489", "NYT19980402.0453", "CNN19980126.1600.1104", "PRI19980115.2000.0186", "PRI19980306.2000.1675", "APW19980227.0494"]        
    print("ndoc:", ndoc_dict)
    print("nrel:", nrel_dict) # NOTE: this is consistant now we the number in PSL paper, using TEPreprocess
    SAVE_DIR = "/data/chengc7/MarkerTRel/tbd_data/relations"
    for type_file in types:
        save_file = os.path.join(SAVE_DIR, f"{type_file}.json")
        json_str = "\n".join([json.dumps(l) for l in final_out[type_file]])
        with open(save_file, 'w') as f:
            f.write(json_str)
            
    # full_test_rel_ids = []
    # for lid, e1_id, e2_id, _, _ in relations.keys():
    #     doc_id = re.search(r"(.*)_\d", lid).group(1)
    #     print(doc_id)
    #     #if doc_id in test_docs:
    #     if 'e' in e1_id.lower() and 'e' in e2_id.lower():
    #         full_test_rel_ids.append([doc_id, e1_id, e2_id])
    #     else:
    #         print(f"event 1: {e1_id}, event 2: {e2_id}")
    # with open(os.path.join(SAVE_DIR, "full_test_rel_ids.pickle"), "wb") as f:
    #     pickle.dump(full_test_rel_ids, f)
        
        
            
            
    
