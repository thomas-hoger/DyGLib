from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, packet_id: np.ndarray=None, attack_type: np.ndarray=None):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.packet_id = packet_id
        self.attack_type = attack_type


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    additionnal_feat_df = pd.read_csv('./processed_data/{}_test/ml_{}_test_additional_labels.csv'.format(dataset_name, dataset_name))

    # Same with test
    graph_df_test = pd.read_csv('./processed_data/{}_test/ml_{}_test.csv'.format(dataset_name, dataset_name))
    edge_raw_features_test = np.load('./processed_data/{}_test/ml_{}_test.npy'.format(dataset_name, dataset_name))
    node_raw_features_test = np.load('./processed_data/{}_test/ml_{}_test_node.npy'.format(dataset_name, dataset_name))
    additionnal_feat_df_test = pd.read_csv('./processed_data/{}_test/ml_{}_test_additional_labels.csv'.format(dataset_name, dataset_name))

    # Concatenate both train and
    edge_raw_features = np.concatenate([edge_raw_features, edge_raw_features_test], axis=0)
    node_raw_features = np.concatenate([node_raw_features, node_raw_features_test], axis=0)

    # Train 
    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    packet_ids = additionnal_feat_df.packet_id.values
    attack_type = additionnal_feat_df.attack_type.values
    
    # Test
    src_node_ids_test = graph_df_test.u.values.astype(np.longlong)
    dst_node_ids_test = graph_df_test.i.values.astype(np.longlong)
    node_interact_times_test = graph_df_test.ts.values.astype(np.float64)
    edge_ids_test = graph_df_test.idx.values.astype(np.longlong)
    labels_test = graph_df_test.label.values
    packet_ids_test = additionnal_feat_df_test.packet_id.values
    attack_type_test = additionnal_feat_df_test.attack_type.values
    
    # Full data (train + test)
    src_node_ids_full = np.concatenate([src_node_ids, src_node_ids_test], axis=0)
    dst_node_ids_full = np.concatenate([dst_node_ids, dst_node_ids_test], axis=0)
    node_interact_times_full = np.concatenate([node_interact_times, node_interact_times_test], axis=0)
    edge_ids_full = np.concatenate([edge_ids, edge_ids_test], axis=0)
    labels_full = np.concatenate([labels, labels_test], axis=0)

    full_data = Data(src_node_ids=src_node_ids_full, dst_node_ids=dst_node_ids_full, node_interact_times=node_interact_times_full, edge_ids=edge_ids_full, labels=labels_full)

    # the setting of seed follows previous works
    random.seed(2020)
    
    train_data = Data(src_node_ids=src_node_ids, 
                      dst_node_ids=dst_node_ids,
                      node_interact_times=node_interact_times,
                      edge_ids=edge_ids, 
                      labels=labels,
                      packet_id=packet_ids,
                      attack_type=attack_type
                      )
    
    test_data = Data(src_node_ids=src_node_ids_test,
                     dst_node_ids=dst_node_ids_test,
                     node_interact_times=node_interact_times_test,
                     edge_ids=edge_ids_test, 
                     labels=labels_test,
                     packet_id=packet_ids_test,
                     attack_type=attack_type_test
                     )


    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, test_data


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data
