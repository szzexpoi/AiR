"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import sys
sys.path.append('./ban')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from attention import BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from torch.autograd import Variable


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
    x = x.sum(1).view(batch_size, x.size(2), x.size(3))
    return x

class BanModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse,num_hid):
        super(BanModel, self).__init__()
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h

            atten, _ = logits[:,g,:,:].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))

        return F.softmax(logits,dim=-1), att

def build_ban(num_token, v_dim, num_hid, num_ans, op='', gamma=4, reasoning=False):
    w_emb = WordEmbedding(num_token, 300, .0, op)
    q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    if not reasoning:
        v_att = BiAttention(v_dim, num_hid, num_hid, gamma)
    else:
        v_att = BiAttention(v_dim, num_hid, num_hid, 1)

    # constructing the model
    b_net = []
    q_prj = []
    c_prj = []
    objects = 36  # minimum number of boxes, originally 10
    for i in range(gamma):
        b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, num_ans, .5)
    counter = Counter(objects)
    if not reasoning:
        return BanModel(w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma, num_hid)
    else:
        return BanModel_Reasoning(w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma, num_hid)

class BanModel_Reasoning(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse,num_hid):
        super(BanModel_Reasoning, self).__init__()
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

        self.semantic_rnn = GRU(256)
        self.semantic_q = nn.Linear(num_hid,256)
        self.semantic_pred = nn.Linear(256,9)
        self.semantic_embed = nn.Embedding(num_embeddings=9,embedding_dim=256) # embedding layer for the semantic operations
        self.att_p = nn.Linear(num_hid,num_hid)
        self.att = nn.Linear(num_hid,1)
        self.att_s = nn.Linear(256,num_hid)
        self.att_v = nn.Linear(2048,num_hid)


    def init_hidden_state(self,batch,s_embed=256):
        init_s = torch.zeros(batch,s_embed).cuda()
        return init_s

    def forward(self, v, b, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        ori_q_emb = q_emb
        boxes = b[:,:,:4].transpose(1,2)
        b_emb = [0] * self.glimpse


        s_x = self.init_hidden_state(len(q),256)
        s_h = torch.tanh(self.semantic_q(ori_q_emb.mean(1)))
        v_att = torch.tanh(self.att_v(F.dropout(v,0.25)))
        op = []
        att_mask = []
        q_emb_pool = []

        for g in range(self.glimpse):
            # reasoning attention
            s_h = self.semantic_rnn(s_x,s_h)
            s_x = F.softmax(self.semantic_pred(s_h),dim=-1)
            op.append(s_x)
            s_x = torch.max(s_x,dim=-1)[1]
            s_x = self.semantic_embed(s_x)
            s_att = torch.tanh(self.att_s(s_h)).unsqueeze(1).expand_as(v_att)
            fuse_feat = torch.tanh(self.att_p(torch.mul(s_att,v_att)))
            reason_att = self.att(fuse_feat)
            reason_att = F.softmax(reason_att.view(reason_att.size(0),-1),dim=-1)
            # reason_att = torch.sigmoid(reason_att.view(reason_att.size(0),-1),dim=-1)
            # cur_v = v + torch.mul(v,reason_att.unsqueeze(-1).expand_as(v))
            cur_v = torch.mul(v,reason_att.unsqueeze(-1).expand_as(v))

            # original ban
            att, logits = self.v_att(cur_v, ori_q_emb) # b x g x v x q
            att, logits = att.squeeze(), logits.squeeze()
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att) # b x l x h

            atten, _ = logits.max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)
            q_emb_pool.append(q_emb)
            att_mask.append(reason_att)


        op = torch.cat([_.unsqueeze(1) for _ in op],dim=1)
        att_mask = torch.cat([_.unsqueeze(1) for _ in att_mask],dim=1)
        valid_op = process_lengths(torch.max(op,dim=-1)[1])
        q_emb_pool = torch.cat([_.unsqueeze(1) for _ in q_emb_pool],dim=1)
        q_emb = select_last(q_emb_pool,valid_op)

        logits = self.classifier(q_emb.sum(1))

        return F.softmax(logits,dim=-1), op, att_mask
