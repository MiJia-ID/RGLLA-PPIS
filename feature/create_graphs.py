import pickle
import dgl
import torch
from torch_geometric.utils import dense_to_sparse
from Bio.PDB import PDBParser
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_coor_train(dis_path,query_ids):
#     dis_load = open(dis_path, 'rb')
#     dis_residue = pickle.load(dis_load)
#     dis_residue['4ne1_p'] = dis_residue.pop('4ne1_pp')
#
#     query_ids = query_ids[:573]
#     coors = []
#     for i in query_ids:
#         coor = dis_residue[i]
#         coors.append(coor)
#     return coors
def get_coor_train(dis_path,query_ids):
    dis_load = open(dis_path, 'rb')
    dis_residue = pickle.load(dis_load)
    #dis_residue['4ne1_p'] = dis_residue.pop('4ne1_pp')

    query_ids = query_ids[:324]
    coors = []
    for i in query_ids:
        coor = dis_residue[i]
        coors.append(coor)
    return coors

def get_coor_test(dis_path,query_ids):
    dis_load = open(dis_path, 'rb')
    dis_residue = pickle.load(dis_load)
    query_ids = query_ids[324:]
    coors = []
    for i in query_ids:
        coor = dis_residue[i]
        coors.append(coor)
    return coors
def get_coor_test_71(dis_path,query_ids):
    dis_load = open(dis_path, 'rb')
    dis_residue = pickle.load(dis_load)
    query_ids = query_ids[:71]
    coors = []
    for i in query_ids:
        coor = dis_residue[i]
        coors.append(coor)
    return coors

# use ESM3 predicted structures to construct adj of protein graphs
# 在另一个环境的脚本里，记得拷过来
def get_adj_predicted(pro_ids):
    adjs = []
    save_files_dir = '/home/mijia/egpdi_evolla/af3_data/'
    fpath = save_files_dir +'dataset_dir_train_186_164_test_71/'
    # fpath = save_files_dir + 'dataset_dir_SJT/'
    adj_type = 'adj_SC_17_predicted'
    for i in pro_ids:
        file = fpath + adj_type + '/{}.pkl'.format(i)
        adj_load = open(file, 'rb')
        adj = pickle.load(adj_load)
        adjs.append(adj)

    return adjs
