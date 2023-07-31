import os
import json
import argparse
from run_temp_rel import convert_examples_to_features, compute_f1, simple_accuracy, InputFeatures
from relation.utils import generate_temp_relation_data
#from shared.data_structures import Dataset, evaluate_predictions
import numpy as np

from custom.data_structures import Dataset, evaluate_predictions
from evaluate.closure import evaluation as closure_evaluate
import torch
from const import SEEDS, MARKERS_I2B2, MODELS, CONTEXT_WINDOWS, LRS, SEEDS, TRAIN_BATCH_SIZE
import pandas as pd

from timegraph.eventEvaluation import eventEvaluation
from timegraph.timexEvaluation import timexEvaluation
from timegraph.temporal_evaluation_adapted import evaluate_two_files
import re
import os
import json
import argparse
from timegraph.tlinkEvaluation import tlinkEvaluation
from timegraph.i2b2Evaluation import get_sub_group_analysis_results, get_type_based_results
from collections import Counter, Set
def get_pred(data): 
    """
    from saved pred get preds of relations
    """
    preds = [] 
    for doc in data: 
        for pred_rel, rel in zip(doc['predicted_relations'], doc['relations']):
            pred_rel, rel = pred_rel[0], rel
            label_id = 4
            pred_label = pred_rel[label_id]
            label = rel[label_id]
            preds.append(pred_label)                 
    return preds

def get_probs(data_json): 
    """
    from saved pred get preds of relations
    """
    probs = [] 
    for doc in data_json:   
        for prob in doc['prob']:
           p1 = prob[0][-1]
           probs.append(p1)               
    return probs

def get_marker_types(data_json, marker="ner", SAVE_DIR=None): 
    """
    from saved pred get preds of relations
    """
    marker_dict = {} # change this to E->id # DEBUG: if T is T->admission, discharge, ignore others
    
    # todo change to dict (doc_id, lid)=[marker1_type, marker2_type]
    marker_type_id = 2
    with open(os.path.join(SAVE_DIR, "doc_lid_ee_map.json"), "r") as f:
        doc_lid_ee_map = json.load(f)
    marker_dict_pair = {}
    for doc in data_json:   
        doc_id = doc['doc_key']
        marker_dict[doc_id] = {}
        
        for (_, _, _, _, rel, lid, _, _, _, _, _, _), (p1, p2) in zip(doc["relations"], doc[marker]):
            marker_dict_pair[doc_id, lid] = [p1[marker_type_id], p2[marker_type_id]]   
            # for both of the e's 
            for e, t in zip(doc_lid_ee_map[doc_id][lid], [p1[marker_type_id], p2[marker_type_id]]):
                if 'sectime' in lid.lower():
                    marker_dict[doc_id][e] = t
                    #pass
                else:
                    if e not in marker_dict[doc_id]:
                        marker_dict[doc_id][e] = t
                    else:
                        # test it's already admission or discharge
                        assert marker_dict[doc_id][e] in ['ADMISSION', 'DISCHARGE'] or marker_dict[doc_id][e]==t
        #if doc_id=='113': # one link has duplicate id and lid=='TL85':
                      
    return marker_dict, marker_dict_pair

def read_json(json_file, pred_file=None):
    gold_docs = [json.loads(line) for line in open(json_file)]
#if pred_file is None: # NOTE: alwyas the case
    return gold_docs
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prediction_file', type=str, default=None, required=True)
    # args = parser.parse_args()

    task="i2b2"
    num_train_epochs=10
    test_xml_dir = "preprocess/corpus/i2b2/ground_truth/merged_xml"
    #train_batch_size = 32
    MARKERS_I2B2=["ner"]#, "ner_plus_time", "ner_pos", "etype", "ner_time_pos"]
    CONTEXT_WINDOWS=[2]
    SEEDS=[1, 2, 3, 4, 5]

    RERUN_I2B2_ORIGINAL_EVAL=False
    #RELATION_TYPE =["relations_updated_sec"] #["relations_truncate"]#["relations", "relations_truncate"]
    RELATION_TYPE =["relations_more_link_with_so"]
    SKIP_FOR_DEBUG=False
    TEST_ORDER_SWAP=False
    #marker = "new"
    #context_window=2
    DO_MARKER_BASED=True
    
    for marker in MARKERS_I2B2:
        for model in MODELS:
            for context_window in CONTEXT_WINDOWS:
                for lr in LRS:
                    for seed in SEEDS:
                        #lr_s = str(lr).replace("0", "")
                        for rel_trunc_type in RELATION_TYPE:
                            
                            for train_batch_size in TRAIN_BATCH_SIZE:
                                
                                i2b2_dataset = f"i2b2_data/{rel_trunc_type}"
                                slurm_out_name = f"{marker}_{model}_{context_window}_{lr}_{num_train_epochs}_{train_batch_size}_{seed}"
                                i2b2_rel_model = f"i2b2_models/{rel_trunc_type}/{marker}"
                                output_dir = f"{i2b2_rel_model}/{slurm_out_name}"
                                pred_file = os.path.join(output_dir, "predictions.json")
                                if not os.path.exists(pred_file):
                                    print(f"Warning: pred not created yet: {pred_file}")
                                    continue
                                else:
                                    print(f"Exists: {pred_file}")
                                result_file = os.path.join(output_dir, "result.txt")
                                result_flipped_file = os.path.join(output_dir, "flipped_result.txt")
                                result_gi = os.path.join(output_dir, "result_gi.txt")
                                result_file_sub = os.path.join(output_dir, "result_sub.csv")
                                result_file_link_type = os.path.join(output_dir, "result_link_type.csv")
                                result_file_marker_type = os.path.join(output_dir, "result_marker_type.csv")
                                #result_file_marker_type_count = os.path.join(output_dir, "result_marker_type_count.csv")


                                # TODO: add back
                                # if os.path.exists(result_file) and os.path.exists(result_file_sub) and os.path.exists(result_file_link_type) and os.path.exists(result_file_marker_type) and not RERUN_I2B2_ORIGINAL_EVAL and not TEST_ORDER_SWAP: # os.path.exists(result_gi) 
                                #     print(f"****Result and GI and SUB existed: {result_file}")
                                #     continue
                                
                                
                                
                                # NOTE: saved data filtered down to before augmentation
                                
                                
                                
                                # data = Dataset(pred_file, entity_type=marker)
                                # print(f"len(data (saved pred)): {len(data)}")

                                data = read_json(pred_file)
                                


                                # TODO: add label2id
                                label_list = []
                                if os.path.exists(os.path.join(output_dir, 'label_list.json')):
                                    with open(os.path.join(output_dir, 'label_list.json'), 'r') as f:
                                        label_list = json.load(f)
                                label2id = {label: i for i, label in enumerate(label_list)}
                                id2label = {i: label for i, label in enumerate(label_list)}
                                
                                # NOTE: load data after augmentation to get gold label ids
                                cached_file = f"{i2b2_rel_model}/cached_test_{task}_{marker}_{model}_{context_window}.pkl"
                                eval_features = convert_examples_to_features(None, label2id, type_file="test", load_pred_after_training=True, load_pred_after_training_file=cached_file, max_seq_length=None, tokenizer=None, special_tokens=None, unused_tokens=None,args=None)
                                eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long) # torch.Size([45420])
                                

                                # get unaugmented_ids
                                test_dataset, test_examples, _ = generate_temp_relation_data(os.path.join(i2b2_dataset, "test.json"), entity_type=marker, use_gold=True, context_window=context_window)
                                unaugmented_ids = []
                                unaugmented_examples = []
                                for i, ex in enumerate(test_examples):
                                    if 'flipped' not in ex['lid']:
                                        unaugmented_ids.append(i)
                                        unaugmented_examples.append(ex)
                                
                                # Note: get pred to correct format and get results
                                pred_labels = get_pred(data)
                                nno_rel = sum([l=='no_relation' for l in pred_labels])
                                assert nno_rel==0
                                preds = [label2id[l] for l in pred_labels]
                                

                                print(f"npred: {len(preds)}")
                                print(f"before filter eval_label_ids.shape: {eval_label_ids.shape}")
                                #eval_label_ids = eval_label_ids[unaugmented_ids]
                                # print(f"after filter eval_label_ids.shape: {eval_label_ids.shape}")

                                result = compute_f1(preds, eval_label_ids, e2e_ngold=len(preds))
                                
                                # Note: this is micro averaging where F1=P=R
                                result['accuracy'] = simple_accuracy(np.array(preds), eval_label_ids.numpy())
                                for key in sorted(result.keys()):
                                    print(f"  %s = %s", key, str(result[key]))
                                                                                

                                                                    
                                #if tempeval:
                                #assert len(unaugmented_examples)==len(preds)
                                predicted_xml_dir = os.path.join(output_dir, "predicted_xml")
                                if not os.path.exists(predicted_xml_dir):
                                    os.mkdir(predicted_xml_dir)
                                
                                doc_link_pred = {}
                                doc_link_prob = {}
                                probs = get_probs(data)
                                assert len(probs)==len(pred_labels)
                                for (ex, pred_l, prob_l)in zip(test_examples, pred_labels, probs): # unaugmented_examples
                                    if ex['docid'] not in doc_link_pred:
                                        doc_link_pred[ex['docid']] = {}
                                        doc_link_prob[ex['docid']] = {}
                                    doc_link_pred[ex['docid']][ex['lid']]=pred_l 
                                    doc_link_prob[ex['docid']][ex['lid']]=prob_l
                                # NOTE: write model output to predicted_xml_dir
                                for doc_id in doc_link_pred.keys():
                                    ce = closure_evaluate(doc_id, doc_link_pred[doc_id], doc_link_prob[doc_id], test_xml_dir, predicted_xml_dir)
                                    ce.eval()
                                    
                                
                                
                                
                                
                                    
                                def i2b2_original_eval_for_one_file(doc_link_pred):        
                                        
                                        # python tlinkEvaluation.py [--oc] [--oo] [--cc] goldstandard_xml_filename system_output_xml_filename
                                        gold_xml_file = os.path.join(test_xml_dir, str(doc_id)+'.xml')
                                        predicted_xml_file = os.path.join(predicted_xml_dir, str(doc_id)+'.xml')

                                        # result for one document
                                        os.system(' '.join(["python2 evaluate/i2b2Evaluation.py --tempeval", gold_xml_file, predicted_xml_file]))
                                        print(f"finished_eval:{doc_id}")

                                
                                # if RERUN_I2B2_ORIGINAL_EVAL:
                                #     i2b2_original_eval(doc_link_pred, test_xml_dir, predicted_xml_dir)
                                    
                                # NOTE: system I2B2 evaluation script, also exact same as CTRL-PG
                                if not os.path.exists(result_file) or RERUN_I2B2_ORIGINAL_EVAL:
                                    os.system(' '.join(["python2 evaluate/i2b2Evaluation.py --tempeval", test_xml_dir, predicted_xml_dir]) + ' > ' + result_file)

                                
                                if TEST_ORDER_SWAP:
                                    predicted_xml_dir = os.path.join(output_dir, "predicted_xml_flipped")
                                    flipped_pred_file = os.path.join(output_dir, 'flipped_predictions.json')
                                    flipped_data = read_json(flipped_pred_file)
                                    if not os.path.exists(predicted_xml_dir):
                                        os.mkdir(predicted_xml_dir)
                                    
                                    flipped_doc_link_pred = {}
                                    flipped_doc_link_prob = {}
                                    flipped_probs = get_probs(flipped_data)
                                    flipped_pred_labels = get_pred(flipped_data)
                                    assert len(flipped_probs)==len(flipped_pred_labels)
                                    
                                    def swap_label_order(l):
                                        if l=="BEFORE":
                                            return "AFTER"
                                        if l=="AFTER":
                                            return "BEFORE"
                                        return l
                                    
                                    for (ex, pred_l, prob_l, original_pred, original_prob) in zip(test_examples, flipped_pred_labels, flipped_probs, pred_labels, probs): # unaugmented_examples
                                        if ex['docid'] not in flipped_doc_link_pred:
                                            flipped_doc_link_pred[ex['docid']] = {}
                                            flipped_doc_link_prob[ex['docid']] = {}

                                        if prob_l > original_prob:
                                            swapped_l = swap_label_order(pred_l)
                                            flipped_doc_link_pred[ex['docid']][ex['lid']]=swapped_l
                                            flipped_doc_link_prob[ex['docid']][ex['lid']]=prob_l
                                        else:
                                            flipped_doc_link_pred[ex['docid']][ex['lid']]=original_pred
                                            flipped_doc_link_prob[ex['docid']][ex['lid']]=original_prob
                                        
                                    
                                    # NOTE: write model output to predicted_xml_dir
                                    for doc_id in flipped_doc_link_pred.keys():
                                        ce = closure_evaluate(doc_id, flipped_doc_link_pred[doc_id], flipped_doc_link_prob[doc_id], test_xml_dir, predicted_xml_dir)
                                        ce.eval()

                                    os.system(' '.join(["python2 evaluate/i2b2Evaluation.py --tempeval", test_xml_dir, predicted_xml_dir]) + ' > ' + result_flipped_file)





                                # NOTE: subgroup analysis
                                #SUB_GROUP_ANALYSIS = True
                                if not os.path.exists(result_file_sub)  or RERUN_I2B2_ORIGINAL_EVAL:
                                    sub_results = get_sub_group_analysis_results(test_xml_dir, predicted_xml_dir)
                                    df_sub = pd.DataFrame.from_dict(sub_results, orient='index', columns=['precision', 'recall', 'pr', 'f1'])
                                    df_sub.to_csv(result_file_sub, index=False)
                                
                                
                                # NOTE:  link type based analysis
                                if not os.path.exists(result_file_link_type) or RERUN_I2B2_ORIGINAL_EVAL:
                                    rel_types = ['SEC', 'ET_OTHER', 'ET', 'TT', 'EE']
                                    # get_type_based_results(goldDir, systemDir, rel_types=[], which_type="marker_type", marker_dict=None):
                                    sub_results = get_type_based_results(test_xml_dir, predicted_xml_dir, rel_types, which_type="link_type")
                                    df_sub = pd.DataFrame.from_dict(sub_results, orient='index', columns=['precision', 'recall', 'pr', 'f1'])
                                    df_sub.to_csv(result_file_link_type, index=False)
                                    
                                # TODO: implement marker type based analysis
                                # only calculate for the optimal
                                use_optimal_only = f"{marker}_{model}_{context_window}_{lr}_{num_train_epochs}_{train_batch_size}"=="ner_bert-base-uncased_2_2e-5_10_16"
                                if (not os.path.exists(result_file_marker_type) and DO_MARKER_BASED) or RERUN_I2B2_ORIGINAL_EVAL: #TODO: change True to use_optimal_only once debug is finished
                                    #rel_types = []
                                    marker_dict, marker_dict_pair = get_marker_types(data, marker=marker, SAVE_DIR=i2b2_dataset)
                                    #marker_types = Set(marker_dict.values())
                                    count_pairs_input = [(a,b) for a, b in marker_dict_pair.values()]
                                    marker_types_counter = Counter(count_pairs_input)
                                    
                                    order_insensitive_pairs_count = {}
                                    for pair, n in marker_types_counter.items():
                                        m1, m2 = pair
                                        if pair not in order_insensitive_pairs_count:
                                            if (m2, m1) in order_insensitive_pairs_count:
                                                order_insensitive_pairs_count[(m2, m1)] += n
                                            else:
                                                order_insensitive_pairs_count[pair] = n
                                        else:
                                            print("show not occur")
                                            
                                    marker_types = marker_types_counter.keys()
                                    marker_types_order_insensitive = order_insensitive_pairs_count.keys()
                                    print("n marker types, insensitive marker types", len(marker_types), len(marker_types_order_insensitive))
                                    
                                    # TODO: also do the insensitive version of analysis
                                    # get_type_based_results(goldDir, systemDir, rel_types=[], which_type="marker_type", marker_dict=None):
                                    sub_results = get_type_based_results(test_xml_dir, predicted_xml_dir, list(marker_types), which_type="marker_type", marker_dict=marker_dict)
                                    df_sub = pd.DataFrame.from_dict(sub_results, orient='index', columns=['precision', 'recall', 'pr', 'f1'])
                                    type_count_list = [sum(marker_types_counter.values())] + list(marker_types_counter.values())
                                    df_sub['count'] = type_count_list # todo: check if by type order is correct
                                    
                                    df_sub.to_csv(result_file_marker_type)    

                                    
                                # # TODO: implement global inference
                                
                                # # NOTE: try performing global inference
                                # TRY_GLOBAL_INFERENCE = False
                                # if TRY_GLOBAL_INFERENCE:
                                #     result_gi = os.path.join(output_dir, "result_gi.txt")
                                #     if not os.path.exists(result_gi):
                                        
                                #         # probs = get_probs(data)
                                        
                                #         assert len(probs)==len(preds)
                                #         #continue
                                        
                                #         print(f"**Performing GI for {result_gi}")


                                #         # for one file 
                                #         systemDic, goldDic = {}, {}
                                #         #goldDic=
                                #         tlinkScores = evaluate_two_files(test_xml_dir, predicted_xml_dir,systemDic,goldDic)

                                        
                                        
                                #         def global_inference_procedure(P1_for_T_T, P2_for_rest):
                                #             def get_predicted_labels_P1_for_T_T():
                                #                 pred_P1_for_T_T = []
                                #                 return pred_P1_for_T_T

                                            
                                #             def construct_time_graph(predicted_T_T):
                                #                 time_graph_G = None
                                #                 return time_graph_G
                                            
                                #             def rank_all_other_predictions_P2():
                                #                 ranked_preds_P2 = None
                                #                 return ranked_preds_P2
                                #             def use_time_graph_to_check_conflict(time_graph_G, ranked_preds_P2):
                                #                 conflict = False
                                                
                                #                 return conflict
                                #             # 1. predict temporal relation p1 on all pairs of T-T
                                #             pred_P1_for_T_T = get_predicted_labels_P1_for_T_T()
                                #             # 2. construct time_graph_G based on T_T
                                #             time_graph_G = construct_time_graph(pred_P1_for_T_T)
                                #             # 3. rank rest of link predictions in descending order according to predicted probablities
                                #             ranked_preds_P2 = rank_all_other_predictions_P2()
                                #             # 4. apply timegraph to ranked_preds_p2 for correction
                                            
                                #             for p in ranked_preds_P2:
                                #                 conflict = use_time_graph_to_check_conflict(time_graph_G, ranked_preds_P2, p)
                                #                 if conflict:
                                #                     # drop p
                                #                     pass
                                #                 else:
                                #                     # add age p to G
                                #                     pass
                                                    
                                            
