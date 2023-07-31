source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh && mamba activate py38
python preprocess_i2b2_merged.py

python formate_i2b2_for_bert.py --type_file test.json
python formate_i2b2_for_bert.py --type_file train.json