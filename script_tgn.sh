#!/bin/bash


# expe=('small_no_time' 'small_no_time_pid' 'small_no_time_pid_sampling') 
# 'accuracy' 'mse'
expe=('supervised_0_001_lr' 'supervised_40_neighbor'  'supervised_batch_500') 

for expe_name in "${expe[@]}"; do
    python eval_link_prediction_CTD5G.py --model_name=TGN --expe_name="$expe_name" --time_feat_dim=30 --dataset_name=old_CTD5G
done

# python eval_link_prediction_CTD5G.py --model_name=TGN --expe_name=supervised --time_feat_dim=30 --dataset_name=old_CTD5G