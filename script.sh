#!/bin/bash

# models=('JODIE' 'DyRep' 'TGAT' 'TGN' 'CAWN' 'EdgeBank' 'TCL' 'GraphMixer')
# models=('DyGFormer')

models=('DyGFormer' 'DyRep' 'TGAT' 'TGN')

for model in "${models[@]}"; do
    python train_reconstruction_CTD5G.py --num_epochs=3 --model_name="$model"
done
