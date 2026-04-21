#!/bin/bash


expe=('small_no_time' 'small_no_time_pid' 'small_no_time_pid_sampling' 'small_seed0') 

for expe_name in "${expe[@]}"; do
    python eval_link_prediction_CTD5G.py --model_name=TGN --expe_name="$expe_name" --time_feat_dim=30
done
