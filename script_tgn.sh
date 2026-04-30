#!/bin/bash


# expe=('small_no_time' 'small_no_time_pid' 'small_no_time_pid_sampling') 
expe=('0_001_lr' '40_neighbor' 'accuracy' 'batch_500' 'mse') 

for expe_name in "${expe[@]}"; do
    python eval_link_prediction_CTD5G.py --model_name=TGN --expe_name="$expe_name" --time_feat_dim=30
done

# python eval_link_prediction_CTD5G.py --model_name=TGN --expe_name=supervised --time_feat_dim=30 --dataset_name=old_CTD5G