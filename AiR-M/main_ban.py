import sys
sys.path.append('./util')
sys.path.append('./ban')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from dataloader_ban import Batch_generator, Batch_generator_submission
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from base_model import build_ban

import numpy as np
import cv2
import argparse
import os
import time
import gc
import tensorflow as tf
from loss import get_mask, semantic_loss, cross_entropy, attention_loss_mask_kld
import json

parser = argparse.ArgumentParser(description='BAN-Reasoning Model for GQA')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--anno_dir',type=str, default=None, help='Directory to GQA question')
parser.add_argument('--prep_dir',type=str, default='./processed_data', help='Directory to preprocessed language files')
parser.add_argument('--box_dir',type=str, default=None, help='Directory to bbox position')
parser.add_argument('--img_dir',type=str, default=None, help='Directory to visual features')
parser.add_argument('--checkpoint_dir',type=str, default=None, help='Directory for saving checkpoint')
parser.add_argument('--weights',type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--num_answer',type=int, default=1500, help='Defining number of candidate answer')
parser.add_argument('--epoch',type=int, default=60, help='Defining maximal number of epochs')
parser.add_argument('--lr',type=float, default=1e-3, help='Defining initial learning rate (default: 4e-4)')
parser.add_argument('--batch_size',type=int, default=256, help='Defining batch size for training (default: 256)')
parser.add_argument('--word_size',type=int, default=300, help='Defining size for word embedding (default: 512)')
parser.add_argument('--embedding_size',type=int, default=1024, help='Defining size for embedding (default: 1024)')
parser.add_argument('--clip',type=float, default=0.25, help='Gradient clipping to prevent gradient explode (default: 0.1)')
parser.add_argument('--alpha',type=float, default=0.5, help='Balance factor for attention loss')
parser.add_argument('--beta',type=float, default=0.5, help='Balance factor for sematic loss')
args = parser.parse_args()

# original lr warm-up from BAN
warm_up_schedule = [cur*args.lr for cur in [0.5,1,1.5,2]]

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(warm_up_schedule,optimizer, epoch):
    "adatively adjust lr based on epoch"
    if epoch < len(warm_up_schedule):
        lr = warm_up_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch in range(10,20):
        lr = warm_up_schedule[-1] * (0.25 ** int(float(epoch-10) / 2))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    train_data = Batch_generator(args.num_answer,args.img_dir,args.box_dir,args.anno_dir,args.prep_dir,'train')
    val_data = Batch_generator(args.num_answer,args.img_dir,args.box_dir,args.anno_dir,args.prep_dir,'val')
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    nb_embedding = train_data.nb_embedding
    vocab = train_data.word2idx

    # adopting the original code from BAN authors
    model = build_ban(nb_embedding, 2048, args.embedding_size, args.num_answer, op='', gamma=4, reasoning=True)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6) #originally 0 decay

    def train(iteration):
        model.train()
        avg_ans_loss = 0
        avg_att_loss = 0
        avg_sem_loss = 0

        for batch_idx,(img,bbox,que,ans,op,att) in enumerate(trainloader):
            img, bbox, que, ans, op, att = Variable(img), Variable(bbox), Variable(que), Variable(ans), Variable(op), Variable(att)
            img, bbox, que, ans, op, att = img.cuda(), bbox.cuda(), que.cuda(), ans.cuda(), op.cuda(), att.cuda()
            optimizer.zero_grad()
            #different training objetives
            output, pred_op, pred_att = model(img,bbox,que)
            ans_mask, att_mask = get_mask(op)
            ans_loss = cross_entropy(output,ans)
            att_loss = attention_loss_mask_kld(pred_att,att,att_mask)
            sem_loss = semantic_loss(pred_op,op)
            loss = ans_loss + att_loss*args.alpha*max((1+np.cos(np.pi*(iteration/300000))),0) + args.beta*sem_loss # originally 0.5, 300000
            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()
            avg_ans_loss = (avg_ans_loss*np.maximum(0,batch_idx) + ans_loss.data.cpu().numpy())/(batch_idx+1)
            avg_att_loss = (avg_att_loss*np.maximum(0,batch_idx) + att_loss.data.cpu().numpy())/(batch_idx+1)
            avg_sem_loss = (avg_sem_loss*np.maximum(0,batch_idx) + sem_loss.data.cpu().numpy())/(batch_idx+1)

            if batch_idx%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('answer loss',avg_ans_loss,step=iteration)
                    tf.summary.scalar('step attention loss',avg_att_loss,step=iteration)
                    tf.summary.scalar('semantic loss',avg_sem_loss,step=iteration)

            iteration += 1

        return iteration

    def test(iteration):
        model.eval()
        total_acc = 0
        total_count = 0

        for batch_idx,(img,bbox,que,ans) in enumerate(valloader):
            img, bbox, que, ans = Variable(img), Variable(bbox), Variable(que), Variable(ans)
            img, bbox, que, ans = img.cuda(), bbox.cuda(), que.cuda(), ans.cuda()
            output, op, att = model(img,bbox,que)

            #computing accuracy
            output = output.data.cpu().numpy()
            ans = ans.data.cpu().numpy()
            output = np.argmax(output,axis=-1)
            ans = np.argmax(ans,axis=-1)

            total_acc += np.count_nonzero(output==ans)
            total_count += len(img)

        total_acc = total_acc*100.0/total_count
        with tf_summary_writer.as_default():
            tf.summary.scalar('validation accuracy',total_acc,step=iteration)
        return total_acc


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_acc = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(warm_up_schedule,optimizer, epoch)
        iteration = train(iteration)
        cur_acc = test(iteration)
        #save the best check point and latest checkpoint
        if cur_acc > val_acc:
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
            val_acc = cur_acc
        torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))


def evaluation():
    test_data = Batch_generator(args.num_answer,args.img_dir,args.box_dir,args.anno_dir,args.prep_dir,'test')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    nb_embedding = test_data.nb_embedding
    vocab = test_data.word2idx

    model = build_ban(nb_embedding, 2048, args.embedding_size, args.num_answer, op='', gamma=4, reasoning=True)
    model.load_state_dict(torch.load(args.weights),strict=False)
    model = model.cuda()
    model.eval()

    overall_score = dict()
    count = 0
    incorr = 0
    for i, (img, bbox, que, ans, qid) in enumerate(testloader):
        img, bbox, que = Variable(img), Variable(bbox), Variable(que)
        img, bbox, que = img.cuda(), bbox.cuda(), que.cuda()

        output, op, att = model(img, bbox, que)
        output = output.data.cpu().numpy()
        att = att.data.cpu().numpy()
        ans = ans.numpy()

        # recording the predicted probability
        for j,cur_id in enumerate(qid):
            if np.argmax(output[j],axis=0) != np.argmax(ans[j],axis=0) or np.sum(ans[j])==0:
                incorr += 1
        count += len(output)

    print('The overall accuracy is %f' %(1-incorr/count))


def submission():
    test_data = Batch_generator_submission(args.num_answer,args.img_dir,args.box_dir,args.anno_dir,args.prep_dir)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    nb_embedding = test_data.nb_embedding
    vocab = test_data.word2idx

    idx2ans = dict()
    for k in test_data.top_answer:
        idx2ans[test_data.top_answer[k]] = k

    model = build_ban(nb_embedding, 2048, args.embedding_size, args.num_answer, op='', gamma=4, reasoning=True)
    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    model.eval()

    submission = []

    for i, (img, bbox, que, qid) in enumerate(testloader):
        img, bbox, que = Variable(img), Variable(bbox), Variable(que)
        img, bbox, que = img.cuda(), bbox.cuda(), que.cuda()

        output,op,att = model(img,bbox,que)
        output = output.data.cpu().numpy()

        # recording the predicted probability
        for j,cur_id in enumerate(qid):
            cur_ans = np.argmax(output[j],axis=0)
            cur_ans = idx2ans[cur_ans]
            tmp_res = {"questionId":str(cur_id),"prediction":cur_ans}
            submission.append(tmp_res)

    with open('./submission/submission_ban.json','w') as f:
        json.dump(submission,f)

if args.mode == 'train':
    main()
elif args.mode == 'eval':
    evaluation()
elif args.mode == 'submission':
    submission()
