# model
import torch.optim

from modules_with_edge_features import *
import torch
from native_sparse_attention_pytorch import SparseAttention

class MainModel(nn.Module):
    def __init__(self,dr,lr,nlayers,lamda,alpha,atten_time,nfeats):
        super(MainModel, self).__init__()

        self.drop1 = nn.Dropout(p=dr)
        # self.fc1 = nn.Linear(640*atten_time, 256)  # for attention
        self.fc1_nsa = nn.Linear(640*atten_time, 256)  # for attention

        self.drop2 = nn.Dropout(p=dr)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

        self.rgn_egnn = RGN_EGNN(nlayers=2, nfeat=nfeats, nhidden=512, nclass=1, dropout=dr,
                                 lamda=lamda, alpha=alpha, variant=True, heads=1)
        self.rgn_gcn2 = RGN_GCN(nlayers=nlayers, nfeat=nfeats, nhidden=128, nclass=1,
                                dropout=dr,
                                lamda=lamda, alpha=alpha, variant=True, heads=1)

        self.multihead_attention = nn.ModuleList([Attention_1(hidden_size=512+128, num_attention_heads=8) for _ in range(atten_time)])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=1e-16)

        self.native_sparse_attns = nn.ModuleList([
            SparseAttention(
                dim=512 + 128,
                dim_head=64,
                heads=atten_time,
                sliding_window_size=2,
                compress_block_size=4,
                selection_block_size=4,
                num_selected_blocks=2
            ) for _ in range(atten_time)
        ])


    def forward(self, G, h, x,adj,efeats):

        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)


        fea2 = self.rgn_gcn2(h, adj)
        fea2 = torch.unsqueeze(fea2, dim=0)

        fea1 = self.rgn_egnn(G, h, x,efeats)
        fea1 = torch.unsqueeze(fea1, dim=0)

        fea = torch.cat([fea1,fea2],dim=2)


        # multi-attention
        attention_outputs = []
        for i in range(len(self.multihead_attention)):
            multihead_output, _ = self.multihead_attention[i](fea)
            attention_outputs.append(multihead_output)
        embeddings = torch.cat(attention_outputs, dim=2)

        # #NSA
        # attention_outputs = []
        # for i in range(len(self.native_sparse_attns)):
        #     multihead_output= self.native_sparse_attns[i](fea)
        #     attention_outputs.append(multihead_output)
        # embeddings = torch.cat(attention_outputs, dim=2)

        out = self.drop1(embeddings)
        out = self.fc1_nsa(out)
        out = self.drop2(out)
        out = self.relu1(out)

        out = self.fc2(out)

        return out

class U_MainModel(nn.Module):
    def __init__(self, dr, lr, nlayers, lamda, alpha, atten_time,nfeats):
        super(U_MainModel, self).__init__()

        self.drop1 = nn.Dropout(p=dr)
        #GCNII+EGNN、GCNII、EGNN这里要改（512+128）
        self.fc1 = nn.Linear(512 * atten_time, 256)  # for attention
        #self.fc1 = nn.Linear(640, 256)  # for attention
        self.drop2 = nn.Dropout(p=dr)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

        self.rgn_egnn = U_GCN_EGNN(nlayers=2, nfeat=nfeats, nhidden=512, nclass=1, dropout=dr,
                                 lamda=lamda, alpha=alpha, variant=True, heads=1, attention='MTP')
        self.rgn_gcn2 = RGN_GCN(nlayers=nlayers, nfeat=nfeats, nhidden=128, nclass=1,
                                dropout=dr,
                                lamda=lamda, alpha=alpha, variant=True, heads=1)
        # self.get = GETModelRunner()
        # GCNII+EGNN、GCNII、EGNN这里要改（512+128）
        self.multihead_attention = nn.ModuleList(
            [Attention_1(hidden_size=512, num_attention_heads=8
                         ) for _ in range(atten_time)])

        # self.block = EfficientAdditiveAttnetion().cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-16)
        # self.fc_get = nn.Linear(6205, 128)

    def forward(self, G, h, x, adj, efeats):
        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea1 = self.rgn_egnn(G, h, x, adj, efeats)
        fea1 = torch.unsqueeze(fea1, dim=0)

        # fea2 = self.get(G, h, x,efeats,device)
        # fea2 = torch.unsqueeze(fea2, dim=0)
        # fea2 = self.fc_get(fea2)

        # fea2 = self.rgn_gcn2(h, adj)
        # fea2 = torch.unsqueeze(fea2, dim=0)

        # GCNII+EGNN、GCNII、EGNN这里要改（512+128）
        # fea = torch.cat([fea2], dim=2)
        fea = torch.cat([fea1], dim=2)
        #fea = torch.cat([fea1, fea2], dim=2)
        # embeddings = self.block(fea)

        # gated self-attention
        attention_outputs = []
        for i in range(len(self.multihead_attention)):
            ####################集成学习消融######################
            multihead_output, _ = self.multihead_attention[i](fea)
            attention_outputs.append(multihead_output)
        embeddings = torch.cat(attention_outputs, dim=2)

        out = self.drop1(embeddings)
        ####################集成学习消融######################
        out = self.fc1(out) #dim:256
        out = self.drop2(out)
        out = self.relu1(out)

        out = self.fc2(out)

        return out

    def get_fc1_embedding(self, G, h, x, adj, efeats, device):
        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea1 = self.rgn_egnn(G, h, x, adj, efeats)
        fea1 = torch.unsqueeze(fea1, dim=0)
        fea2 = self.get(G, h, x, efeats, device)
        fea2 = torch.unsqueeze(fea2, dim=0)
        fea2 = self.fc_get(fea2)

        fea = torch.cat([fea1, fea2], dim=2)

        attention_outputs = []
        for i in range(len(self.multihead_attention)):
            multihead_output, _ = self.multihead_attention[i](fea)
            attention_outputs.append(multihead_output)
        embeddings = torch.cat(attention_outputs, dim=2)

        out = self.drop1(embeddings)
        out = self.fc1(out)  # 提取fc1输出

        return out  # 这是fc1的输出

    def get_origin_embedding(self, G, h, x, adj, efeats, device):
        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)
        h = self.fc_get(h)

        return h

    def get_egnn_embedding(self, G, h, x, adj, efeats, device):
        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea1 = self.rgn_egnn(G, h, x, adj, efeats)
        fea1 = torch.unsqueeze(fea1, dim=0)

        return fea1  # 这是fc1的输出

    def get_egnn_get_embedding(self, G, h, x, adj, efeats, device):
        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea1 = self.rgn_egnn(G, h, x, adj, efeats)
        fea1 = torch.unsqueeze(fea1, dim=0)
        fea2 = self.get(G, h, x, efeats, device)
        fea2 = torch.unsqueeze(fea2, dim=0)
        fea2 = self.fc_get(fea2)

        fea = torch.cat([fea1, fea2], dim=2)

        return fea  # 这是fc1的输出

