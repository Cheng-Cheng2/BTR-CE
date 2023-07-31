import os, re
import pandas as pd
from run_eval import get_pred
from const import SEEDS, MODELS, CONTEXT_WINDOWS, LRS, SEEDS, MARKERS_TBD
from custom.data_structures import Dataset
from run_temp_rel import convert_examples_to_features, compute_f1, simple_accuracy, InputFeatures
from relation.utils import generate_temp_relation_data
import torch
import json, pickle
from run_eval import read_json

# parser = argparse.ArgumentParser()
# parser.add_argument('--prediction_file', type=str, default=None, required=True)
# args = parser.parse_args()




# MARKERS_TBD=["etype"]
# MODELS=["bert-base-uncased"]#, "roberta-base"]



def get_tbd_results():
    task="tbd"
    tbd_dataset="tbd_data/relations"

    NUM_TRAIN_EPOCHS=[20]
    NUM_TRAIN_EPOCHS=[8, 16]
    TRAIN_BATCH_EPOCHS=[8, 16]
    relation_type = ["relations", "relations_truncate"]
    LRS=["2e-5"]

    res_doc_dict = {}
    for marker in MARKERS_TBD:
        for model in MODELS:
            for context_window in CONTEXT_WINDOWS:
                for lr in LRS:
                    for seed in SEEDS:
                        for num_train_epochs in NUM_TRAIN_EPOCHS:
                            for train_batch_size in TRAIN_BATCH_EPOCHS:
                                for rel_trunc_type in relation_type:
                                    
                                    #lr_s = str(lr).replace("0", "")
                                    slurm_out_name = f"{marker}_{model}_{context_window}_{lr}_{num_train_epochs}_{train_batch_size}_{seed}"
                                    dict_out_name = f"{marker}_{model}_{context_window}_{lr}_{num_train_epochs}_{train_batch_size}"
                                    tbd_rel_model = f"tbd_models/{rel_trunc_type}/{marker}"
                                    output_dir = f"{tbd_rel_model}/{slurm_out_name}"
                                    pred_file = os.path.join(output_dir, "predictions.json")

                                    
                                    #result_file = os.path.join(output_dir, "result.txt")
                                    if os.path.exists(pred_file):

                                        # NOTE: saved data filtered down to before augmentation
                                        #data = Dataset(pred_file, entity_type=marker)
                                        data = read_json(pred_file)
                                        print(f"len(data (saved pred)): {len(data)}")

                                        # TODO: add label2id
                                        label_list = []
                                        if os.path.exists(os.path.join(output_dir, 'label_list.json')):
                                            with open(os.path.join(output_dir, 'label_list.json'), 'r') as f:
                                                label_list = json.load(f)
                                        label2id = {label: i for i, label in enumerate(label_list)}
                                        id2label = {i: label for i, label in enumerate(label_list)}
                                        
                                        # NOTE: load data after augmentation to get gold label ids
                                        cached_file = f"{tbd_rel_model}/cached_test_{task}_{marker}_{model}_{context_window}.pkl"
                                        eval_features = convert_examples_to_features(None, label2id, type_file="test", load_pred_after_training=True, load_pred_after_training_file=cached_file, max_seq_length=None, tokenizer=None, special_tokens=None, unused_tokens=None,args=None)
                                        eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long) # torch.Size([45420])
                                        

                                            # NOTE: get unaugmented_ids
                                        test_dataset, test_examples, _ = generate_temp_relation_data(os.path.join(tbd_dataset, "test.json"), entity_type=marker, use_gold=True, context_window=context_window)
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
                                        eval_label_ids = eval_label_ids[unaugmented_ids]
                                        print(f"after filter eval_label_ids.shape: {eval_label_ids.shape}")


                                            
                                        result = compute_f1(preds, eval_label_ids, e2e_ngold=len(preds))
                                        result['seed']=seed
                                        result['method']=slurm_out_name
                                        res_doc_dict[slurm_out_name] = result 
                                    
    df = pd.DataFrame.from_dict(res_doc_dict, orient="index").sort_values(by='f1')    
    vals = ['precision', 'recall', 'f1', 'seed', 'method']
    df = df[vals]
    df.sort_values(by='f1')
    df = df.groupby('method').mean().sort_values(by='f1')
    
    df.to_csv("agg_tbd.csv")
    return df


                                
if __name__=='__main__':                                   
    get_tbd_results()                            