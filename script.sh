#!/bin/bash

models=('TGN' 'DyRep')

logfile="execution_times.log"
: > "$logfile"   # Vide le fichier au début

for model in "${models[@]}"; do
    echo "===== $model =====" | tee -a "$logfile"

    /usr/bin/time -f "Temps réel: %E\nTemps CPU: %U user %S sys\nMémoire max: %M KB" \
        -o "$logfile" -a \
        # python eval_reconstruction_CTD5G.py  --model_name="$model" --expe_name=supervised --dataset_name=old_CTD5G --time_feat_dim=30
        python eval_reconstruction_CTD5G.py  --model_name="$model" --expe_name="$model"_seed0 --dataset_name=CTD5G_fasttext_retrain --time_feat_dim=30

    echo "" >> "$logfile"
done