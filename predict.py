import os, torch, argparse, pickle, warnings
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import dgl
from dgl import DGLGraph
from torch_geometric.utils import dense_to_sparse
from feature.create_node_feature_71 import create_dataset
from feature.create_graphs import get_coor_test_71,get_adj_predicted
from feature.create_edge import create_dis_matrix
from model_with_edge_features import U_MainModel, MainModel
import numpy as np

warnings.filterwarnings("ignore")
seed_value = 1995
torch.manual_seed(seed_value)
cuda_index = 3
device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() and cuda_index < torch.cuda.device_count() else "cpu")

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/mijia/egpdi_evolla/af3_data/')
parser.add_argument("--edgefeats_path", type=str, default='/home/mijia/egpdi_evolla/af3_data/dataset_dir_SJT/SJT1/edge_features/EdgeFeats_predicted_SC_17_181.pkl')
parser.add_argument("--model_path", type=str, default='/home/mijia/egpdi_evolla/Model/AF3_SC_egnn_gcn2_multihead/')
args = parser.parse_args()

# ==== 载入数据部分 ====
root_dir = args.dataset_path
test_path = root_dir + 'SJT1.txt'
pkl_path = root_dir + 'dataset_dir_SJT/' + 'PPIS_residue_feas_PSA.pkl'
esm2_5120_path = root_dir + 'esm2/'
dis_path = root_dir + 'dataset_dir_SJT/' + 'PPIS_psepos_SC.pkl'
ProtTrans_path = root_dir + 'prottrans/'


query_ids, sequences, labels = [], [], []
with open(test_path, 'r') as f:
    test_text = f.readlines()
    for i in range(0, len(test_text), 3):
        query_ids.append(test_text[i].strip()[1:])
        sequences.append(test_text[i + 1].strip())

X, y = create_dataset(query_ids, test_path, test_path, test_path, pkl_path, esm2_5120_path, ProtTrans_path,
                      residue=True, one_hot=True, esm_5120=True, prottrans=True)
distance_matrixs = create_dis_matrix(dis_path, query_ids)
features = X
coors = get_coor_test_71(dis_path, query_ids)
adjs = get_adj_predicted(query_ids)

graphs = []
for adj in adjs:
    edge_index, _ = dense_to_sparse(adj)
    G = dgl.graph((edge_index[0], edge_index[1])).to(device)
    graphs.append(G)

with open(args.edgefeats_path, 'rb') as f:
    efeats = pickle.load(f)

dataframe = pd.DataFrame({
    "ID": query_ids,
    "sequence": sequences,
    "features": features,
    "coors": coors,
    "adj": adjs,
    "graph": graphs,
    "efeats": efeats
})

# ==== 推理数据集类 ====
class PredictDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (row["ID"], row["sequence"], row["features"], row["graph"], row["efeats"], row["adj"], row["coors"])

def predict_collate(samples):
    name_batch, seq_batch, node_features_batch, graph_batch, efeat_batch, adj_batch, coors_batch = map(list, zip(*samples))
    graph_batch = dgl.batch(graph_batch)
    return name_batch, seq_batch, node_features_batch, graph_batch, efeat_batch, adj_batch, coors_batch

# ==== 推理函数 ====
def predict_prob_only(model, data_loader):
    model.eval()
    results = []

    with torch.no_grad():
        for names, seqs, node_features_batch, graph_batch, efeat_batch, adj_batch, coors_batch in data_loader:
            protein_id = names[0]
            sequence = seqs[0]

            node_features_batch = torch.tensor(node_features_batch[0], dtype=torch.float32)
            coors_batch = torch.tensor(coors_batch[0], dtype=torch.float32)
            adj_batch = torch.tensor(adj_batch[0], dtype=torch.float32)
            efeat_batch = torch.tensor(efeat_batch[0], dtype=torch.float32)

            if torch.cuda.is_available():
                node_features_batch = node_features_batch.cuda()
                coors_batch = coors_batch.cuda()
                adj_batch = adj_batch.cuda()
                efeat_batch = efeat_batch.cuda()
                graph_batch = graph_batch.to(torch.device("cuda"))

            y_pred = model(graph_batch, node_features_batch, coors_batch, adj_batch, efeat_batch)
            logits = y_pred.cpu().numpy().squeeze()

            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            elif logits.ndim != 1:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            for i in range(len(sequence)):
                residue = sequence[i]
                logit = float(logits[i])
                prob = 1 / (1 + np.exp(-logit))  # sigmoid 函数将 logit 映射到 [0, 1]
                results.append([protein_id, i, residue, logit, prob])

    pred_df = pd.DataFrame(results, columns=["Protein_ID", "Residue_Index", "Residue", "Logit", "Sigmoid_Prob"])
    return pred_df


# ==== 主执行函数 ====
def predict_129(model_path, fold_id):
    model = MainModel(dr=0.3, lr=0.0001, nlayers=4, lamda=1.1, alpha=0.1, atten_time=8, nfeats=41 + 20 + 1024 + 5120).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    test_dataset = PredictDataset(dataframe)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=predict_collate)

    pred_df = predict_prob_only(model, test_loader)

    pred_df.to_csv(f"Fold{i}_Pred_Probabilities.csv", index=False)

    # 读取原始数据
    file_path = f"Fold{i}_Pred_Probabilities.csv"
    df = pd.read_csv(file_path)

    # 添加三个标记列
    df['High_Prob_Flag'] = (df['Sigmoid_Prob'] > 0.99995).astype(int)
    df['Logit_gt_10'] = (df['Logit'] > 10).astype(int)
    df['Sigmoid_Prob_gt_0.99995'] = (df['Sigmoid_Prob'] > 0.99995).astype(int)

    # 保存为新的 CSV 文件
    df.to_csv(f"Fold{i}_Pred_Probabilities.csv", index=False)


# ==== 多模型执行 ====
model_paths = [
    args.model_path + f'Fold{i}predicted_edgeFeats_best_AUPR_model.pkl'
    for i in range(1, 6)
]

for i, path in enumerate(model_paths, start=1):
    print(f"Running inference for Fold {i}")
    predict_129(path, i)
