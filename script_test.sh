python train_link_prediction_CTD5G.py --num_epochs=1 --model_name=GraphMixer --expe_name=lp_baseline --time_feat_dim=3 
python train_reconstruction_CTD5G.py --num_epochs=3 --model_name=GraphMixer --expe_name=ae_baseline --time_feat_dim=3 

python eval_reconstruction_CTD5G.py --model_name=GraphMixer --expe_name=ae_baseline --time_feat_dim=3
python eval_link_prediction_CTD5G.py --model_name=GraphMixer --expe_name=lp_baseline --time_feat_dim=3