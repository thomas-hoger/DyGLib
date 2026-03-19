import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

cmap = plt.cm.RdYlGn_r
feature_vocab = {'http2.path': 0, 'pfcp.seid': 1, 'pfcp.f_teid.teid': 2, 'gtp.ext_hdr.pdu_ses_con.pdu_type': 3, 'http2.ausfInstanceId': 4, 'http2.ueId': 5, 'http2.imsi': 6, 'http2.amfInstanceId': 7, 'http2.cause': 8, 'http2.service-names': 9, 'http2.target-nf-type': 10, 'pfcp.apply_action.forw': 11, 'ip_src': 12, 'pfcp.msg_type': 13, 'pfcp.outer_hdr_creation.teid': 14, 'http2.nfInstanceId': 15, 'http2.target-nf-instance-id': 16, 'http2.supis': 17, 'gtp.teid': 18, 'pfcp.f_teid.ipv4_addr': 19, 'http2.ipv4Address': 20, 'pfcp.ue_ip_addr_ipv4': 21, 'http2.nfType': 22, 'http2.request.supi': 23, 'pfcp.pdr_id': 24, 'pfcp.apply_action.drop': 25, 'http2.supi': 26, 'http2.subscriberIdentifier': 27, 'ip.ip_dst': 28, 'pfcp.far_id': 29, 'ip.ip_src': 30, 'http2.jwt.scope': 31, 'ip_dst': 32, 'http2.nf-type': 33, 'pfcp.cause': 34, 'pfcp.outer_hdr_creation.ipv4': 35, 'http2.requester-nf-type': 36, 'pfcp.f_seid.ipv4': 37, 'http2.jwt.sub': 38, 'http2.method': 39, 'http2.servingNfId': 40}
attack_type_vocab = {'user_traffic': 0, 'modify_drop': 1, 'flood_etablishment': 2, 'applicative_scan': 3, 'seid_fuzzing': 4, 'pfcp_in_gtp': 5, 'uplink_wake_random_u': 6, 'set_random_ue_idle': 7, 'cn_mitm': 8, 'add_random_nf': 9, 'remove_random_nf': 10, 'flood_deletion': 11, 'uplink_spoofing': 12, 'fuzz': 13, 'downlink_wake_random': 14, 'modify_dupl': 15, 'deregister_random_ue': 16, 'register_random_ue': 17, 'restart': 18, 'unknown': 19}

loss_min = 0
loss_max = 1 

def plot_graph_with_loss(nx_graph, node_colors, node_labels, save_dir):
    pos = nx.spring_layout(nx_graph, seed=42)

    edge_colors = [nx_graph[u][v]['color'] for u, v in nx_graph.edges()]
    node_color_list = [node_colors.get(n, "blue") for n in nx_graph.nodes()]

    plt.figure(figsize=(16, 14))

    nx.draw(
        nx_graph,
        pos,
        with_labels=False,
        edge_color=edge_colors,
        node_color=node_color_list,
        node_size=50,
        width=3,
    )

    # 🔥 labels uniquement pour sources
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        labels=node_labels,
        font_size=6,
        font_color="gray"
    )

    # 🔥 garder seulement les edges avec forte loss
    edge_labels = {
        (u, v): data["label"]
        for u, v, data in nx_graph.edges(data=True)
        if data.get("loss", 0) > 0.2
    }

    nx.draw_networkx_edge_labels(
        nx_graph,
        pos,
        edge_labels=edge_labels,
        font_size=6
    )

    plt.title("Graph with Loss Gradient (Green → Red)")
    plt.axis("off")
    plt.savefig(save_dir)
    print(f"Saved graph to {save_dir}")


def evaluate_model_reconstruction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000, df_labels: pd.DataFrame = None):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()
    
    attack_counters = {}

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses = []
        evaluate_embeddings = []
        
        evaluate_metrics = []
        
        
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_label, batch_attack, batch_packet_id = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                evaluate_data.labels[evaluate_data_indices], evaluate_data.attack_type[evaluate_data_indices], evaluate_data.packet_id[evaluate_data_indices]

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_pids=batch_packet_id,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      node_pids=batch_packet_id,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      node_pids=batch_packet_id,
                                                                      time_gap=time_gap)

            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      node_pids=batch_packet_id)

            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            
            selected_attack = None
            for att_type in np.unique(batch_attack):
                
                if att_type not in attack_counters:
                    attack_counters[att_type] = 0
                
                if attack_counters[att_type] < 3:
                    attack_counters[att_type] += 1
                    selected_attack = att_type
                    break 
               
            if not selected_attack:
                continue 

            nx_graph = nx.Graph()
            node_colors = {}
            node_labels = {}
            
            mask = batch_attack == selected_attack
            selected_attack_label = list(attack_type_vocab.keys())[selected_attack]
            
            original_msg = model[0].edge_raw_features[batch_edge_ids]
            event_embedding = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            
            for i in range(len(event_embedding)):
                src = batch_src_node_ids[mask][i].item()
                dst = batch_dst_node_ids[mask][i].item()

                edge_text = list(feature_vocab.keys())[original_msg[mask][i].argmax().item()]

                loss_masked = loss_func(event_embedding[mask][i], original_msg[mask][i]).item()
                norm_loss = (loss_masked - loss_min) / (loss_max - loss_min)
                loss_masked = max(0, min(1, norm_loss))
                
                color = cmap(loss_masked)

                if batch_label[mask][i].item() == 1:
                    node_colors[src] = "red"
                else:
                    node_colors[src] = "blue"
                node_colors[dst] = "black"

                node_labels[src] = selected_attack_label

                nx_graph.add_edge(batch_src_node_ids[mask][i].item(), batch_dst_node_ids[mask][i].item(), label=edge_text, color=color, loss=loss_masked)

                loss = loss_func(event_embedding[i], original_msg[i]).item()
                evaluate_losses.append(loss)
                evaluate_embeddings.append(event_embedding[i])
                
            save_dir = f'./figures/{model_name}/{selected_attack_label}_{attack_counters[selected_attack]}.png'
            os.makedirs(f'./figures/{model_name}', exist_ok=True)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss}, {len(attack_counters)}/{len(attack_type_vocab)} attack types')
            
            if attack_counters[selected_attack] < 3:
                plot_graph_with_loss(nx_graph, node_colors, node_labels, save_dir)


    os.makedirs('./losses', exist_ok=True)
    np.save(f'./losses/{model_name}.npy', np.array(evaluate_losses))
        
    os.makedirs('./embeddings', exist_ok=True)
    np.save(f'./embeddings/{model_name}.npy', np.array(evaluate_embeddings)) 
      
    return evaluate_losses, evaluate_metrics

def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
            labels = torch.from_numpy(batch_labels).float().to(predicts.device)

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                              batch_src_node_ids=batch_src_node_ids,
                                                                                              batch_dst_node_ids=batch_dst_node_ids,
                                                                                              current_batch_start_time=batch_node_interact_times[0],
                                                                                              current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=args.test_ratio)

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}'for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
