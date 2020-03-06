import numpy as np
import random
import os
import time
import operator
import torch
import torch.utils.data as data
import json
import cv2
import gc

class Batch_generator(data.Dataset):
    def __init__(self,nb_answer,ori_img,img_dir,box_dir,que_dir,prep_dir,mode='train'):
        self.mode = mode
        self.ori_img = ori_img
        self.img_dir = img_dir
        self.nb_answer = nb_answer
        self.box_dir = box_dir
        # selecting top answers
        self.top_answer = json.load(open(os.path.join(prep_dir,'ans2idx_1500.json')))
        self.word2idx = json.load(open(os.path.join(prep_dir,'word2idx_1500.json')))

        if self.mode == 'train':
            self.semantic = np.load(os.path.join(prep_dir,'semantic_mask.npy'),allow_pickle=True).item()
            self.op2idx = json.load(open(os.path.join(prep_dir,'op2idx.json')))

        if not mode == 'test':
            self.question = json.load(open(os.path.join(que_dir,mode+'_balanced_questions.json')))
        else:
            self.question = json.load(open(os.path.join(que_dir,'testdev_balanced_questions.json')))

        self.nb_embedding = len(self.word2idx.keys())
        self.Q, self.Img, self.answer, self.Qid = self.init_data()

    def init_data(self,):
        question = []
        answer = []
        imgid = []
        Qid = []

        for qid in self.question.keys():
            cur_A = self.question[qid]['answer']
            if cur_A in self.top_answer:
                tmp_A = np.zeros([self.nb_answer,]).astype('float32')
                tmp_A[self.top_answer[cur_A]] = 1
                cur_A = tmp_A
            elif self.mode != 'test':
                continue
            else:
                cur_A = np.zeros([self.nb_answer,]).astype('float32')

            cur_Q = self.question[qid]['question']
            cur_Q = cur_Q.replace('?','').replace('.','').replace(',',' ')
            cur_Q = cur_Q.split(' ')

            if len(cur_Q)>18 and self.mode != 'test': #remove questions that exceed specific length, originally 14
                continue

            if self.mode == 'train' and len(self.semantic[qid])>4: # remove question with too complicated reasoning
                continue

            cur_Q = convert_idx(cur_Q,self.word2idx)
            cur_img = self.question[qid]['imageId']

            question.append(cur_Q)
            answer.append(cur_A)
            imgid.append(cur_img)
            Qid.append(qid)

        return question, imgid, answer, Qid

    def __getitem__(self,index):
        max_len = 18 if self.mode != 'test' else 25 # originally 14
        question = self.Q[index]
        answer = self.answer[index]
        img_id = self.Img[index]
        raw_img = cv2.imread(os.path.join(self.ori_img,str(img_id)+'.jpg'))
        h, w, c = raw_img.shape

        # padding question if necessary
        if len(question) < max_len:
            pad = np.zeros(max_len-len(question))
            question = np.concatenate((question,pad),axis=0)
        question = np.array(question).astype('int')

        answer = np.array(answer).astype('float32')

        # load image features
        img = np.load(os.path.join(self.img_dir,str(img_id)+'.npy'))
        bbox = np.load(os.path.join(self.box_dir,str(img_id)+'.npy'))

        # normalize the position of bbox
        for i in range(len(bbox)):
            bbox[i,0] /= w
            bbox[i,1] /= h
            bbox[i,2] /= w
            bbox[i,3] /= h

        if self.mode == 'test':
            return img, bbox, question, answer, self.Qid[index]
        elif self.mode == 'train':
            semantic = self.semantic[self.Qid[index]]
            op = [self.op2idx[cur[0]] for cur in semantic]
            att_mask = [cur[1] for cur in semantic]
            while len(op)<4: # padding the semantics to 4
                op.append(0)
                att_mask.append((np.ones([len(att_mask[0]),])*1.0/len(att_mask[0])).astype('float32'))

            att_mask = np.array(att_mask).astype('float32')
            op_mask = np.zeros([len(op),9]).astype('float32')
            for i in range(len(op_mask)):
                op_mask[i,op[i]] = 1
                att_mask[i] /= np.sum(att_mask[i])+1e-16

            return img, bbox, question, answer, op_mask, att_mask
        else:
            return img, bbox, question, answer

    def __len__(self,):
        return len(self.Img)


class Batch_generator_submission(data.Dataset):
    def __init__(self,nb_answer,ori_img,img_dir,box_dir,que_dir,lang_dir):
        self.ori_img = ori_img
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.nb_answer = nb_answer
        self.top_answer = json.load(open(os.path.join(lang_dir,'ans2idx_1500.json')))
        self.word2idx = json.load(open(os.path.join(lang_dir,'word2idx_1500.json')))
        self.question = json.load(open(os.path.join(que_dir,'submission_all_questions.json')))

        self.nb_embedding = len(self.word2idx.keys())
        self.Q, self.Img, self.Qid = self.init_data()

    def init_data(self,):
        question = []
        imgid = []
        Qid = []

        for qid in self.question.keys():
            cur_Q = self.question[qid]['question']
            cur_Q = cur_Q.replace('?','').replace('.','').replace(',',' ')
            cur_Q = cur_Q.split(' ')
            cur_Q = convert_idx(cur_Q,self.word2idx)
            cur_img = self.question[qid]['imageId']

            question.append(cur_Q)
            imgid.append(cur_img)
            Qid.append(qid)

        return question, imgid, Qid

    def __getitem__(self,index):
        max_len = 30
        question = self.Q[index]
        img_id = self.Img[index]
        raw_img = cv2.imread(os.path.join(self.ori_img,str(img_id)+'.jpg'))
        h, w, c = raw_img.shape

        # padding question if necessary
        if len(question) < max_len:
            pad = np.zeros(max_len-len(question))
            question = np.concatenate((question,pad),axis=0)
        question = np.array(question).astype('int')

        # load image features
        img = np.load(os.path.join(self.img_dir,str(img_id)+'.npy'))
        bbox = np.load(os.path.join(self.box_dir,str(img_id)+'.npy'))

        # normalize the position of bbox
        for i in range(len(bbox)):
            bbox[i,0] /= w
            bbox[i,1] /= h
            bbox[i,2] /= w
            bbox[i,3] /= h
        return img, bbox, question, self.Qid[index]

    def __len__(self,):
        return len(self.Img)

#convert words within string to index
def convert_idx(sentence,word2idx):
    idx = []
    for word in sentence:
        if word in word2idx:
            idx.append(word2idx[word])
        else:
            idx.append(word2idx['UNK'])

    return idx
