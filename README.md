# BTR-CE
This code repository is for our paper "Typed Markers and Context for clinical for Clinical Temporal Relation Extraction"

## Data preprocess
refer to the `readme.sh` in `preprocess` for running the data preprocessing pipeline for i2b2.

## Train the model
To reproduce the best result for I2B2 in the paper run:
```
task=i2b2
relations_type=relations_more_link_with_so
marker=ner
seed=1
num_train_epochs=10
context_window=2
lr=1e-2
train_batch_size=16

i2b2_dataset=../i2b2_data/${relations_type}
i2b2_rel_model=../i2b2_models/${relations_type}/$marker
output_dir=$i2b2_rel_model/$slurm_out_name

mkdir -p $output_dir

python run_temp_rel.py \
    --task $task \
    --marker $marker \
    --do_train \
    --do_eval \
    --eval_test \
    --model $model \
    --do_lower_case \
    --context_window ${context_window} \
    --max_seq_length 512 \
    --data_dir $i2b2_dataset \
    --output_dir $output_dir \
    --use_cached \
    --num_train_epochs ${num_train_epochs} \
    --seed ${seed} \
    --learning_rate ${lr} \
    --eval_batch_size 32 \
    --train_batch_size $train_batch_size \
    --eval_metric i2b2_f1
```

# Evaluation
run `run_eval.py` to aggregate results across experiments and visualize them in `aggregate_results.ipynb.

# Citation
We used code from `https://github.com/princeton-nlp/PURE` to build our model.
