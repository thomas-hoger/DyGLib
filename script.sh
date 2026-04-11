#!/bin/bash

# models=('JODIE' 'DyRep' 'TGAT' 'TGN' 'CAWN' 'TCL' 'GraphMixer')
# models=('JODIE' 'DyRep' 'DyGFormer' 'TGN' 'TCL' 'GraphMixer')

models=('DyGFormer' 'GraphMixer' 'TGAT' 'TCL' 'CAWN' 'TGN' 'JODIE' 'DyRep') 

for model in "${models[@]}"; do
    # python train_reconstruction_CTD5G.py --num_epochs=2 --model_name="$model"
    python eval_model_CTD5G.py --model_name="$model"
done
