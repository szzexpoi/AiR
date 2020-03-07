import json
import numpy as np
import os
import cv2
import operator
from glob import glob
import re
import argparse

parser = argparse.ArgumentParser(description="Generating attention mask for supervision")
parser.add_argument("--bbox_dir", type=str, required=True, help="path to object bounding box")
parser.add_argument("--question", type=str, required=True, help="path to GQA question")
parser.add_argument("--scene_graph", type=str, required=True, help="path to GQA scene graph")
parser.add_argument("--semantics", type=str, default='./data', help="path to processed semantics")
parser.add_argument("--save", type=str, default='./AiR-M/processed_data', help="path for saving the data")
args = parser.parse_args()


bbox_dir = args.bbox_dir
question = json.load(open(os.path.join(args.question,'train_balanced_questions.json')))
ori_scene_graph = json.load(open(os.path.join(args.scene_graph,'train_sceneGraphs.json')))
semantic_info = json.load(open(os.path.join(args.semantics,'simplified_semantics_train.json')))

# IoU for computing the atention
def overlap(mask, obj):
    dx = min(mask[2], obj[0]+obj[2]) - max(mask[0], obj[0])
    dy = min(mask[3], obj[1]+obj[3]) - max(mask[1], obj[1])
    if (dx>=0) and (dy>=0):
        return dx*dy/((obj[2]*obj[3])+(mask[2]-mask[0])*(mask[3]-mask[1])-dx*dy)
    else:
        return 0

# use the coverage of co-exist objects for attentions if the ground truth is a non-existing objects
def prior_coverage(bbox,scene_graph,prior_appearance,obj,mask):
    prior_threshold = 3 # focus on the top-k co-appeared objects
    if obj not in prior_appearance:
        return mask # new object or object entangled with attributes, leave it for now
    count = 0
    flag = True
    for i in range(len(prior_appearance[obj])):
        cur_name = prior_appearance[obj][i][0]
        for cur_obj in scene_graph:
            if scene_graph[cur_obj]['name'] == cur_name:
                obj_mask = (scene_graph[cur_obj]['x'],scene_graph[cur_obj]['y'],scene_graph[cur_obj]['w'],scene_graph[cur_obj]['h'])
                for j in range(len(bbox)):
                    mask[j] += overlap(bbox[j],obj_mask)/prior_threshold
                count += 1
            if count >= prior_threshold:
                flag = False
                break
        if not flag:
            break

    return mask

# compute the co-existence of objects
def get_prior_appearance():
    # computing the prior information about object co-appearance
    prior_appearance = dict()
    prior_scene_graph = ori_scene_graph
    for cur_img in prior_scene_graph.keys():
        objects = prior_scene_graph[cur_img]['objects']
        for i in range(len(objects)):
            cur_obj = objects[list(objects.keys())[i]]['name']
            if cur_obj not in prior_appearance:
                prior_appearance[cur_obj] = dict()
            tmp_pool = []
            for j in range(len(objects)):
                co_obj = objects[list(objects.keys())[j]]['name']
                if cur_obj== co_obj:
                    continue
                if co_obj not in prior_appearance[cur_obj]:
                    prior_appearance[cur_obj][co_obj] = 0
                if co_obj not in tmp_pool:
                    prior_appearance[cur_obj][co_obj] += 1
                    tmp_pool.append(co_obj) # computing the co-appearance only once for each scene

    # sorting the co-appearance
    for cur_obj in prior_appearance.keys():
        cur_appearance = prior_appearance[cur_obj]
        cur_appearance = sorted(cur_appearance.items(), key=operator.itemgetter(1))
        prior_appearance[cur_obj] = cur_appearance[::-1]

    return prior_appearance

def main():
    result = dict()
    prior_appearance = get_prior_appearance()
    invalid_count = 0
    total_count = 0
    for qid in question.keys():
        cur_que = question[qid]['question']
        img_id = question[qid]['imageId']
        cur_scene_graph = ori_scene_graph[img_id]['objects']
        cur_bbox = np.load(os.path.join(bbox_dir,str(img_id)+'.npy'))
        tmp_res = []

        for i in range(len(semantic_info[qid])):
            cur_op = semantic_info[qid][i][0]
            cur_mask = np.zeros([len(cur_bbox),])
            for j in range(1,3):
                if semantic_info[qid][i][j] is None:
                    continue
                elif 'obj:' in semantic_info[qid][i][j]:
                    cur_obj = semantic_info[qid][i][j][4:]
                    cur_obj = cur_obj.split(',')
                    for obj in cur_obj:
                        obj_mask = (cur_scene_graph[obj]['x'],cur_scene_graph[obj]['y'],cur_scene_graph[obj]['w'],cur_scene_graph[obj]['h'])
                        for k in range(len(cur_bbox)):
                            cur_mask[k] += overlap(cur_bbox[k],obj_mask)
                    # for some cases none of the regional bboxes cover the objects of interest
                    if np.sum(cur_mask) == 0:
                        obj = cur_scene_graph[cur_obj[0]]['name']
                        cur_mask = prior_coverage(cur_bbox,cur_scene_graph,prior_appearance,obj,cur_mask)

                elif 'obj*' in semantic_info[qid][i][j]:
                    cur_obj = semantic_info[qid][i][j][5:].split(',')
                    obj_pool = []
                    for obj in cur_obj:
                        obj = obj.split(' ')[0]
                        if obj not in obj_pool:
                            cur_mask = prior_coverage(cur_bbox,cur_scene_graph,prior_appearance,obj,cur_mask)
                            obj_pool.append(obj)

            # if the regional bboxes do not even match any relevant objects
            if np.sum(cur_mask) == 0:
                cur_mask = np.ones([len(cur_bbox),])*1.0/len(cur_bbox)
                invalid_count += 1

            tmp_res.append((cur_op,cur_mask))
            total_count += 1

        result[qid] = tmp_res

    print('Invalid count: %f' %(invalid_count/total_count))
    np.save(os.path.join(args.save,'semantic_mask'),result)

main()
