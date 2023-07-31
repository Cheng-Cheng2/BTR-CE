import os, re
import pandas as pd

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prediction_file', type=str, default=None, required=True)
    # args = parser.parse_args()

    task="i2b2"
    num_train_epochs=10
    i2b2_dataset="i2b2_data/relations"
    test_xml_dir = "preprocess/corpus/i2b2/ground_truth/merged_xml"

    MARKERS_I2B2=["ner", "etype", "ner_plus_time"]
    MODELS=["bert-base-uncased"]
    CONTEXT_WINDOWS=[0]
    LRS=["1e-5", "2e-5", "1e-4"]
    SEEDS=[1]
    
    # MARKERS_I2B2=["etype"]
    # MODELS=["bert-base-uncased"]
    # CONTEXT_WINDOWS=[0]
    # LRS=[1e-5]
    # SEEDS=[1]
    
    res_doc_dict = {}
    for marker in MARKERS_I2B2:
        for model in MODELS:
            for context_window in CONTEXT_WINDOWS:
                for lr in LRS:
                    for seed in SEEDS:
                        #lr_s = str(lr).replace("0", "")
                        slurm_out_name = f"{marker}_{model}_{context_window}_{lr}_{num_train_epochs}_{seed}"
                        i2b2_rel_model = f"i2b2_models/relations/{marker}"
                        output_dir = f"{i2b2_rel_model}/{slurm_out_name}"
                        
                        
                        result_file = os.path.join(output_dir, "result.txt")
                        
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
                                res_doc_dict[slurm_out_name] = res_dict
                                
print(f"n res: len(res_doc_dict)")
df = pd.DataFrame.from_dict(res_doc_dict)       
df                             
                                
                                    
                        