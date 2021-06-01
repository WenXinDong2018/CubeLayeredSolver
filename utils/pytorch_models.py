import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def getModel(name):
    if name == "singleH":
        return ResnetModelBaseline
    if name == "multiH_baseline":
        return ResnetModelBaseline
    if name == "multiH_4plus1":
        return ResnetModelFC
    if name == "multiH_4plus2":
        return ResnetModel4_2
    if name == "multiH_3plus2":
        return ResnetModelRES
    else:
        raise ValueError("model name (%s)does not exit" % update_method)

'''This file contains 5 pytorch models'''

class ResnetModelBaseline(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm
        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x


'''3+2 model'''
class ResnetModelRES(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks-1): #one less block
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out1 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])
        self.fc_out2 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])
        self.fc_out3 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])

        self.out_l1 = nn.Linear(resnet_dim, 1)
        self.out_l2 = nn.Linear(resnet_dim, 1)
        self.out_l3 = nn.Linear(resnet_dim, 1)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks-1):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        l1 = self.fc_out1[0](x)
        l1 = self.fc_out1[1](l1)
        l1 = F.relu(l1)
        l1 = self.fc_out1[2](l1)
        l1 = self.fc_out1[3](l1)

        l2 = self.fc_out2[0](x)
        l2 = self.fc_out2[1](l2)
        l2 = F.relu(l2)
        l2 = self.fc_out2[2](l2)
        l2 = self.fc_out2[3](l2)

        l3 = self.fc_out3[0](x)
        l3 = self.fc_out3[1](l3)
        l3 = F.relu(l3)
        l3 = self.fc_out3[2](l3)
        l3 = self.fc_out3[3](l3)


        l1 = self.out_l1(l1)
        l2 = self.out_l2(l2)
        l3 = self.out_l3(l3)
        final = torch.stack((l1, l2, l3)).squeeze().T
        return final


'''4+1 model'''
class ResnetModelFC(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks): #one less block
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out1 = nn.Linear(resnet_dim, resnet_dim)
        self.fc_out2 = nn.Linear(resnet_dim, resnet_dim)
        self.fc_out3 = nn.Linear(resnet_dim, resnet_dim)
        self.out_l1 = nn.Linear(resnet_dim, 1)
        self.out_l2 = nn.Linear(resnet_dim, 1)
        self.out_l3 = nn.Linear(resnet_dim, 1)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        l1 = self.fc_out1(x)
        l2 = self.fc_out2(x)
        l3 = self.fc_out3(x)
        l1 = self.out_l1(l1)
        l2 = self.out_l2(l2)
        l3 = self.out_l3(l3)

        final = torch.stack((l1, l2, l3)).squeeze().T
        return final



'''4+2 block model'''
class ResnetModel4_2(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out1 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])
        self.fc_out2 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])
        self.fc_out3 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])

        self.out_l1 = nn.Linear(resnet_dim, 1)
        self.out_l2 = nn.Linear(resnet_dim, 1)
        self.out_l3 = nn.Linear(resnet_dim, 1)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        l1 = self.fc_out1[0](x)
        l1 = self.fc_out1[1](l1)
        l1 = F.relu(l1)
        l1 = self.fc_out1[2](l1)
        l1 = self.fc_out1[3](l1)

        l2 = self.fc_out2[0](x)
        l2 = self.fc_out2[1](l2)
        l2 = F.relu(l2)
        l2 = self.fc_out2[2](l2)
        l2 = self.fc_out2[3](l2)

        l3 = self.fc_out3[0](x)
        l3 = self.fc_out3[1](l3)
        l3 = F.relu(l3)
        l3 = self.fc_out3[2](l3)
        l3 = self.fc_out3[3](l3)


        l1 = self.out_l1(l1)
        l2 = self.out_l2(l2)
        l3 = self.out_l3(l3)
        final = torch.stack((l1, l2, l3)).squeeze().T
        return final

'''Transformer based model'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


'''transformer based'''
class ResnetModelTransformer(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool, num_heads:int = 1):
        super().__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.pos_encoder = PositionalEncoding(one_hot_depth)
        encoder_layers = TransformerEncoderLayer(one_hot_depth, num_heads, dim_feedforward=h1_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, resnet_dim)
        self.bn1 = nn.BatchNorm1d(resnet_dim)
        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_fc1 = nn.Linear(resnet_dim, resnet_dim)
            res_bn1 = nn.BatchNorm1d(resnet_dim)
            res_fc2 = nn.Linear(resnet_dim, resnet_dim)
            res_bn2 = nn.BatchNorm1d(resnet_dim)
            self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))



        self.fc_out1 = nn.Linear(resnet_dim, resnet_dim)
        self.fc_out2 = nn.Linear(resnet_dim, resnet_dim)
        self.fc_out3 = nn.Linear(resnet_dim, resnet_dim)
        self.out_l1 = nn.Linear(resnet_dim, 1)
        self.out_l2 = nn.Linear(resnet_dim, 1)
        self.out_l3 = nn.Linear(resnet_dim, 1)

    def forward(self, states_nnet):
        x = states_nnet
        x = F.one_hot(x.long(), self.one_hot_depth)
        x = x.float()

        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        x = output.view(-1, self.state_dim * self.one_hot_depth)
        x=self.fc1(x)
        x = self.bn1(x)

        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)



        l1 = self.fc_out1(x)
        l2 = self.fc_out2(x)
        l3 = self.fc_out3(x)
        l1 = self.out_l1(l1)
        l2 = self.out_l2(l2)
        l3 = self.out_l3(l3)



        final = torch.stack((l1, l2, l3)).squeeze().T
        return final

