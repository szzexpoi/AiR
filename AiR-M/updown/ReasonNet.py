import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from skipthoughts import BayesianUniSkip
from gru import BayesianGRU
from scipy.sparse import random as sprand
import numpy as np
from transformers import XLNetModel, BertModel


def L2_Norm(input,dim=2,eps=1e-12):
    return input/input.norm(p=2, dim=dim).clamp(min=eps).expand_as(input)

def process_lengths(input):
    """
    Computing the lengths of sentences in current batchs
    """
    max_length = input.size(1)
    lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
    return lengths

def select_last(x, lengths):
    """
    Adaptively select the hidden state at the end of sentences
    """
    batch_size = x.size(0)
    seq_length = x.size(1)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        mask[i][lengths[i]-1].fill_(1)
    mask = Variable(mask)
    x = x.mul(mask)
    x = x.sum(1).view(batch_size, x.size(2))
    return x

def select_att(x,lengths):
    batch_size = x.size(0)
    seq_length = x.size(1)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        mask[i][:lengths[i]].fill_(1)
    mask = Variable(mask)
    x = x.mul(mask)
    x = x.sum(1).view(batch_size, x.size(2))
    return x/(x.sum(-1,keepdim=True).expand_as(x)+1e-16)


class GRU(nn.Module):
    """
    Gated Recurrent Unit without long-term memory
    """
    def __init__(self,embed_size=512):
        super(GRU,self).__init__()
        self.update_x = nn.Linear(embed_size,embed_size,bias=True)
        self.update_h = nn.Linear(embed_size,embed_size,bias=True)
        self.reset_x = nn.Linear(embed_size,embed_size,bias=True)
        self.reset_h = nn.Linear(embed_size,embed_size,bias=True)
        self.memory_x = nn.Linear(embed_size,embed_size,bias=True)
        self.memory_h = nn.Linear(embed_size,embed_size,bias=True)

    def forward(self,x,state):
        z = F.sigmoid(self.update_x(x) + self.update_h(state))
        r = F.sigmoid(self.reset_x(x) + self.reset_h(state))
        mem = F.tanh(self.memory_x(x) + self.memory_h(torch.mul(r,state)))
        state = torch.mul(1-z,state) + torch.mul(z,mem)
        return state

class ReasonNet(nn.Module):
    """
    ReasonNet Model for GQA
    """
    def __init__(self,word_size=620,rnn='GRU',nb_embedding=None,dropout=0.3,img_size=2048,q_size=512,s_size=256,embedding_size=1024,skip_thoughts=False,vocab=None,nb_answer=2000):
        super(ReasonNet, self).__init__()
        self.rnn = rnn
        self.img_size = img_size
        self.q_size = q_size
        self.skip_thoughts = skip_thoughts
        self.embedding_size = embedding_size
        self.s_size = s_size

        if not self.skip_thoughts:
            print('Using standard RNN')
            self.word_embedding = nn.Embedding(num_embeddings=nb_embedding,embedding_dim=word_size,padding_idx=0)
            if self.rnn == 'LSTM':
                self.q_model = nn.LSTM(input_size=word_size,hidden_size=q_size,num_layers=1,batch_first=True,bias=True,dropout=0.25)
            elif self.rnn == 'GRU':
                self.q_model = BayesianGRU(input_size=word_size, hidden_size=q_size, dropout=0.25)
            else:
                assert self.rnn in ['LSTM','GRU'], 'Selected language model not implemented'
        else:
            print('Using Skip-thoughts Bayesian GRU')
            self.q_model = BayesianUniSkip(dir_st='./Skip-thoughts-pretrained' ,vocab=vocab)

        # module for different step
        self.semantic_rnn = GRU(self.s_size)
        self.semantic_q = nn.Linear(self.q_size,self.s_size)
        self.semantic_pred = nn.Linear(self.s_size,9)
        self.semantic_embed = nn.Embedding(num_embeddings=9,embedding_dim=self.s_size) # embedding layer for the semantic operations
        self.att_v = nn.Linear(self.img_size,embedding_size) #1200
        self.att_p = nn.Linear(embedding_size,embedding_size)
        self.att = nn.Linear(embedding_size,1)
        self.att_v_drop = nn.Dropout(dropout)
        self.att_s = nn.Linear(self.s_size,embedding_size)

        self.v_fc = nn.Linear(self.img_size,embedding_size)
        self.q_fc = nn.Linear(self.q_size,embedding_size)
        self.fc = nn.Linear(embedding_size,nb_answer)
        self.v_drop = nn.Dropout(dropout)

        self.fc_drop = nn.Dropout(dropout)
        self.q_drop = nn.Dropout(dropout)

    def init_hidden_state(self,batch,s_embed=256,v_embed=1024):
        init_s = torch.zeros(batch,s_embed).cuda()
        init_v = torch.zeros(batch,v_embed).cuda()
        return init_s,init_v

    def freeze_semantic(self,):
        for module in [self.semantic_rnn,self.semantic_q,self.semantic_pred,self.semantic_embed]:
            for pare in module.parameters():
                pare.requires_grad = False

    def forward(self,img_feat,que,gt_op=None,ss_rate=2):
        #processing question features for attention computation
        lengths = process_lengths(que)
        if not self.skip_thoughts:
            que = self.word_embedding(que) #removed tanh activation
            if self.rnn == 'LSTM':
                output, (ht,ct) = self.q_model(que)
                q = select_last(output, lengths)
            elif self.rnn == 'GRU':
                output, ht = self.q_model(que)
                q = select_last(output, lengths)
        else:
            q = self.q_model(que)

        s_x, v_h = self.init_hidden_state(len(q),self.s_size,self.embedding_size)
        op = []
        att_mask = []
        step_weight = []

        s_h = torch.tanh(self.semantic_q(q))
        v_att = torch.tanh(self.att_v(self.att_v_drop(img_feat)))
        nb_step = 4

        for i in range(nb_step):
            # predict the reasoning operation
            s_h = self.semantic_rnn(s_x,s_h)
            s_x = F.softmax(self.semantic_pred(s_h),dim=-1)
            op.append(s_x)
            ss_prob = torch.rand(1)
            if ss_prob>(1-ss_rate):
                s_x = torch.max(s_x,dim=-1)[1]
            else:
                s_x = torch.max(gt_op[:,i],dim=-1)[1]
            s_x = self.semantic_embed(s_x)

            # compute the attention by considering the reasoning operation
            s_att = torch.tanh(self.att_s(s_h)).unsqueeze(1).expand_as(v_att)
            fuse_feat = torch.tanh(self.att_p(torch.mul(s_att,v_att)))
            soft_att = self.att(fuse_feat)
            soft_att = F.softmax(soft_att.view(soft_att.size(0),-1),dim=-1)
            att_mask.append(soft_att)

        op = torch.cat([_.unsqueeze(1) for _ in op],dim=1)
        valid_op = process_lengths(torch.max(op,dim=-1)[1])
        att_mask = torch.cat([_.unsqueeze(1) for _ in att_mask],dim=1)

        soft_att = select_att(att_mask,valid_op).unsqueeze(-1)
        img_feat = torch.mul(img_feat,soft_att.expand_as(img_feat))
        img_feat = img_feat.sum(1)

        x_v = torch.tanh(self.v_fc(self.v_drop(img_feat)))
        x_q = torch.tanh(self.q_fc(self.q_drop(q)))
        x = torch.mul(x_v,x_q)
        x = self.fc(self.fc_drop(x))
        x = F.softmax(x,dim=-1)

        return x, op, att_mask # for training
        # return x, op, soft_att.squeeze() # for testing or agg_training
