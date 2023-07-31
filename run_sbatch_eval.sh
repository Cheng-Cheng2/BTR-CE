jobname=eval
#gpu=p100
#ngpu=4


slurm_out_name=eval_job
printf "slurm_out_name: ${slurm_out_name}\n"
sbatch --mem=16g \
    --job-name=${jobname} \
    --output=out/${task}_${slurm_out_name}_gpu:${gpu}:${ngpu}_%j.out \
    --export=seed=${seed},lr=${lr},context_window=${context_window},num_train_epochs=${num_train_epochs},slurm_out_name=${slurm_out_name},model=${model},marker=${marker},slurm_out_file=${slurm_out_name}_gpu:${gpu}:${ngpu},task=${task},train_batch_size=${train_batch_size}\
    --time=20:00:00 \
    rel_eval.sh