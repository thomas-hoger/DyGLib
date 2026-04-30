#!/bin/bash

# models=('DyGFormer' 'GraphMixer' 'TGAT' 'TCL' 'CAWN' 'TGN' 'JODIE' 'DyRep') 

models=('GraphMixer' 'TGAT' 'TCL' 'CAWN' 'TGN' 'JODIE' 'DyRep') 

for model in "${models[@]}"; do
    # python train_reconstruction_CTD5G.py --num_epochs=2 --model_name="$model"
    # python train_link_prediction_CTD5G.py --num_epochs=5 --model_name="$model" --time_feat_dim=30 --expe_name=baseline
    python eval_link_prediction_CTD5G.py --model_name="$model" --time_feat_dim=30 --expe_name=baseline
done
