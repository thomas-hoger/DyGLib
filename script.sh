#!/bin/bash

models=('DyGFormer' 'GraphMixer' 'TCL' 'CAWN' 'TGN' 'JODIE' 'DyRep')

logfile="execution_times.log"
: > "$logfile"   # Vide le fichier au début

for model in "${models[@]}"; do
    echo "===== $model =====" | tee -a "$logfile"

    /usr/bin/time -f "Temps réel: %E\nTemps CPU: %U user %S sys\nMémoire max: %M KB" \
        -o "$logfile" -a \
        python eval_model_CTD5G.py --model_name="$model"

    echo "" >> "$logfile"
done