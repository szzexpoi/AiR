import torch
import torch.nn.functional as F
from torch.autograd import Variable

epsilon = 1e-16

def answer_loss(pred_ans,gt_ans,mask):
    gt_ans = gt_ans.unsqueeze(1).expand_as(pred_ans)
    loss = -(gt_ans*torch.log(torch.clamp(pred_ans,min=epsilon,max=1))).sum(-1)
    loss = torch.mul(loss,mask).sum(-1)/mask.sum(-1)
    return loss.mean()

def cross_entropy(input_,target):
    input_ = input_.view(input_.size(0), -1)
    loss = -(target*torch.log(torch.clamp(input_,min=epsilon,max=1))).sum(-1)
    return loss.mean()

def get_mask(gt_op):
    max_length = gt_op.size(1)
    gt_op = gt_op.sum(-1)
    lengths = list(max_length - gt_op.data.eq(0).sum(1).squeeze()) # get the valid length
    att_mask = gt_op.data.new().resize_as_(gt_op.data).fill_(0)
    ans_mask = gt_op.data.new().resize_as_(gt_op.data).fill_(0)
    for i in range(len(gt_op)):
        att_mask[i][:lengths[i]].fill_(1)
        ans_mask[i][:lengths[i]].fill_(1)
        # ans_mask[i][lengths[i]-1].fill_(1)
    att_mask = Variable(att_mask).cuda()
    ans_mask = Variable(ans_mask).cuda()

    return ans_mask, att_mask

def semantic_loss(pred_op,gt_op):
    loss = -(gt_op*torch.log(torch.clamp(pred_op,min=epsilon,max=1))).sum(-1)
    return loss.mean()

def attention_loss_mask(pred_att,gt_att,mask):
    # loss = ((pred_att-gt_att)**2).sum(-1) # MSE
    loss = -(gt_att*torch.log(torch.clamp(pred_att,min=epsilon,max=1))).sum(-1) # CE
    loss = torch.mul(loss,mask).mean()
    return loss

def attention_loss_mask_kld(pred_att,gt_att,mask):
    pred_att = pred_att.view(pred_att.size(0),pred_att.size(1),-1)
    gt_att = gt_att.view(gt_att.size(0),gt_att.size(1),-1)
    loss = torch.mul(gt_att,torch.log(torch.div(gt_att,pred_att+epsilon) + epsilon))
    loss = loss.sum(-1)
    loss = torch.mul(loss,mask).mean()
    return loss        

def attention_loss(pred_att,gt_att):
    # loss = ((pred_att-gt_att)**2).sum(-1) # MSE
    loss = -(gt_att*torch.log(torch.clamp(pred_att,min=epsilon,max=1))).sum(-1) # CE
    return loss.mean()    

def kld(pred_att,gt_att):
    pred_att = pred_att.view(pred_att.size(0),-1)
    gt_att = gt_att.view(gt_att.size(0),-1)
    loss = torch.mul(gt_att,torch.log(torch.div(gt_att,pred_att+epsilon) + epsilon))
    loss = loss.sum(-1)
    return torch.mean(loss)