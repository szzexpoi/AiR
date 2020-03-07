import json
import numpy as np
import os
import cv2
import operator
from glob import glob
import re
import operator
import argparse

parser = argparse.ArgumentParser(description="Computing the AiR-E scores for attention")
parser.add_argument("--image", type=str, required=True, help="path to GQA images")
parser.add_argument("--question", type=str, required=True, help="path to GQA question")
parser.add_argument("--scene_graph", type=str, required=True, help="path to GQA scene graph")
parser.add_argument("--att_dir", type=str, required=True, help="path to attention")
parser.add_argument("--bbox_dir", type=str, default=None, help="path to object bounding box (for object-based attention)")
parser.add_argument("--semantics", type=str, default='./data', help="path to processed semantics")
parser.add_argument("--att_type", type=str, default='human', help="specify the type of attention (human, spatial, object)")
parser.add_argument("--save", type=str, default='./data', help="path for saving the data")
args = parser.parse_args()

# computing the prior information about object co-appearance
def get_prior_appearance():
    prior_appearance = dict()
    prior_scene_graph = json.load(open(os.path.join(args.scene_graph,'train_sceneGraphs.json')))
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

# use the coverage of co-exist objects for attentions if the ground truth is a non-existing objects
def prior_coverage(obj_prob,scene_graph,prior_appearance,obj):
    prior_threshold = 20 # focus on the top-k co-appeared objects
    if obj not in prior_appearance:
        return 1 # new object or object entangled with attributes, leave it for now
    prior_prob = []
    for i in range(min(prior_threshold,len(prior_appearance[obj]))):
        co_obj = prior_appearance[obj][i][0]
        for tmp_obj in obj_prob.keys():
            if tmp_obj == 'background':
                continue
            if scene_graph[tmp_obj]['name'] == co_obj:
                prior_prob.append(obj_prob[tmp_obj])
    return np.mean(prior_prob) if len(prior_prob)>0 else 0

# compute the aggregated score for each bounding box
def overlap(mask, obj):
    return mask[obj[1]:(obj[1]+obj[3]),obj[0]:(obj[0]+obj[2])].mean()

# filtering operations
def filter_attribute(attr,obj,w,h):
    if attr not in ['bottom','top','left','right','middle']:
        return attr in obj['attributes']
    elif attr == 'bottom':
        return (obj['y']+obj['h']/2)>(h/2)
    elif attr == 'top':
        return (obj['y']+obj['h']/2)<=(h/2)
    elif attr == 'left':
        return (obj['x']+obj['w']/2)<=(w/2)
    elif attr == 'right':
        return (obj['x']+obj['w']/2)>(w/2)
    else: # middle
        return np.abs(obj['x']-(w/2))<=(w/4) and np.abs(obj['y']-(h/2)) <=(h/4)

# project object-based attention to spatial map for evaluation
def box2spatial(att,bbox,res):
    for i in range(len(att)):
        res[int(bbox[i][1]):int(bbox[i][3]),int(bbox[i][0]):int(bbox[i][2])] += att[i]
    return res

def main():
    img_dir = args.image
    question = json.load(open(os.path.join(args.question,'val_balanced_questions.json')))
    scene_graph = json.load(open(os.path.join(args.scene_graph,'val_sceneGraphs.json')))
    semantic_info = json.load(open(os.path.join(args.semantics,'simplified_semantics_val.json')))
    bbox_dir = args.bbox_dir
    prior_appearance = get_prior_appearance()

    if args.att_type == 'human':
        # human attention is stored as saliency map
        att_map = glob(os.path.join(args.att_dir,'*.png'))
        valid_qid = [os.path.basename(cur)[:-4] for cur in att_map]
    else:
        # model attention is assumed to be stored in a numpy dictionary, with key being the qid and value the map
        att_map = np.load(args.att_dir,allow_pickle=True).item()
        valid_qid = list(att_map.keys())

    # main loop for evaluation
    final_result = dict()
    for qid in valid_qid:
        img_id = question[qid]['imageId']
        img = cv2.imread(os.path.join(img_dir,str(img_id)+'.jpg'))
        h, w, _ = img.shape

        if args.att_type == 'human':
            cur_att = cv2.imread(os.path.join(args.att_dir,str(qid)+'.png'))[:,:,0].astype('float32') # for human attention
        elif args.att_type == 'spatial':
            cur_att = att_map[qid].reshape([7,7]) # for spatial attention
        else:
            cur_bbox = np.load(os.path.join(bbox_dir,str(img_id)+'.npy'))
            cur_att = box2spatial(att_map[qid],cur_bbox,np.zeros([h,w])) # for object-based attention

        cur_att = cv2.resize(cur_att,(w,h))
        cur_att = (cur_att-np.mean(cur_att))/(np.std(cur_att)+1e-16)
        cur_que = question[qid]['question']
        cur_scene_graph = scene_graph[img_id]['objects']

        # loop through all object bboxs (scene graph) and input regions (features) to compute the overlap
        obj_prob = dict()
        background_mask = np.zeros([h,w])
        for cur_obj in cur_scene_graph:
            obj_mask = (cur_scene_graph[cur_obj]['x'],cur_scene_graph[cur_obj]['y'],cur_scene_graph[cur_obj]['w'],cur_scene_graph[cur_obj]['h'])
            obj_prob[cur_obj] = overlap(cur_att,obj_mask)
            background_mask[obj_mask[1]:(obj_mask[1]+obj_mask[3]),obj_mask[0]:(obj_mask[0]+obj_mask[2])] = 1
        background_att = (cur_att*(1-background_mask)).mean()

        if len(obj_prob) == 0 or np.isnan(np.max(list(obj_prob.values()))):
            continue

        obj_prob['background'] = background_att  # take into account the background

        # computing the AiR-E for different operations
        semantic_prob = []
        for i in range(len(semantic_info[qid])):
            tmp_semantic = dict()
            tmp_semantic['candidate'] = dict()
            tmp_semantic['dependency'] = []
            cur_op = semantic_info[qid][i][0]

            if cur_op == 'select': # select object of a specific category
                if 'obj:' in semantic_info[qid][i][1]:
                    cur_obj = semantic_info[qid][i][1][4:]
                    cur_obj = cur_obj.split(',')
                    name_pool = []
                    for obj in cur_obj:
                        name_pool.append(cur_scene_graph[obj]['name'])
                    for obj in obj_prob.keys():
                        if obj == 'background':
                            continue
                        if cur_scene_graph[obj]['name'] in name_pool:
                            tmp_semantic['candidate'][obj] = obj_prob[obj]
                    tmp_semantic['score'] = np.max(list(tmp_semantic['candidate'].values()))
                else: # no object of the category exists in the scene, use prior coverage
                    cur_obj = semantic_info[qid][i][1][5:].split(',')
                    obj = cur_obj[0].split(' ')[0]
                    tmp_semantic['score'] = prior_coverage(obj_prob,cur_scene_graph,prior_appearance,obj)
            elif cur_op in ['verify','query']:
                ref_semantic = semantic_prob[semantic_info[qid][i][-1][0]]
                tmp_semantic['dependency'].append(semantic_info[qid][i][-1][0])
                tmp_semantic['candidate'] = ref_semantic['candidate']
                if 'scene' in semantic_info[qid][i][2]:
                    tmp_semantic['score'] = 0.5
                elif 'obj*:' in semantic_info[qid][i][2]:
                    query_obj = semantic_info[qid][i][2][5:]
                    query_obj = query_obj.split(',')[0].split(' ')[0]
                    tmp_semantic['score'] = prior_coverage(obj_prob,cur_scene_graph,prior_appearance,obj)
                else:
                    query_obj = semantic_info[qid][i][2][4:]
                    query_obj = query_obj.split(',')[0]
                    query_obj_name = cur_scene_graph[query_obj]['name']
                    query_pool = []
                    for obj in tmp_semantic['candidate'].keys():
                        if cur_scene_graph[obj]['name'] == query_obj_name:
                            query_pool.append(obj_prob[obj])
                    if len(query_pool)>0:
                        tmp_semantic['score'] = np.max(query_pool)
                    else:
                        tmp_semantic['score'] = 0
            elif cur_op == 'compare':
                tmp_semantic['score'] = []
                for ref_idx in semantic_info[qid][i][-1]:
                    tmp_semantic['dependency'].append(ref_idx)
                    ref_semantic = semantic_prob[ref_idx]
                    tmp_semantic['score'].append(np.max(list(ref_semantic['candidate'].values())))
                    tmp_semantic['candidate'].update(ref_semantic['candidate'])
                tmp_semantic['score'] = np.mean(tmp_semantic['score'])
            elif cur_op == 'relate':
                # reference object
                ref_obj_pool = dict()
                ref_score = 1
                for ref_idx in semantic_info[qid][i][-1]:
                    tmp_semantic['dependency'].append(ref_idx)
                    ref_semantic = semantic_prob[ref_idx]
                    ref_obj_pool.update(ref_semantic['candidate'])
                    ref_score *= ref_semantic['score']
                if 'obj:' in semantic_info[qid][i][1]:
                    cur_obj = semantic_info[qid][i][1][4:]
                    cur_obj = cur_obj.split(',')[0]
                    cur_obj_name = cur_scene_graph[cur_obj]['name']
                    for obj in obj_prob.keys():
                        if obj == 'background':
                            continue
                        if cur_scene_graph[obj]['name'] == cur_obj_name:
                            tmp_semantic['candidate'][obj] = obj_prob[obj]
                    tmp_semantic['score'] = np.max(list(tmp_semantic['candidate'].values()))
                else: # no object of the category exists in the scene, use prior coverage
                    cur_obj = semantic_info[qid][i][1][5:].split(',')
                    obj = cur_obj[0].split(' ')[0]
                    tmp_semantic['score'] = prior_coverage(obj_prob,cur_scene_graph,prior_appearance,obj)
                if len(ref_obj_pool) > 0:
                    tmp_semantic['score'] = np.mean([np.max(list(ref_obj_pool.values())),tmp_semantic['score']])
                else:
                    tmp_semantic['score'] = np.mean([ref_score,tmp_semantic['score']])
                tmp_semantic['candidate'].update(ref_obj_pool)
            elif cur_op == 'filter':
                ref_semantic = semantic_prob[semantic_info[qid][i][-1][0]]
                tmp_semantic['dependency'].append(semantic_info[qid][i][-1][0])
                attr = semantic_info[qid][i][1]
                inv_logic = False
                if 'not' in attr:
                    attr = re.search(r'\(([^)]+)\)', attr).group(1)
                    inv_logic = True
                for obj in ref_semantic['candidate'].keys():
                    if filter_attribute(attr, cur_scene_graph[obj],w,h) and not inv_logic:
                        tmp_semantic['candidate'][obj] = obj_prob[obj]
                    elif not filter_attribute(attr, cur_scene_graph[obj],w,h) and inv_logic:
                        tmp_semantic['candidate'][obj] = obj_prob[obj]

                if len(list(tmp_semantic['candidate'].values())) > 0:
                    tmp_semantic['score'] = np.max(list(tmp_semantic['candidate'].values()))
                else:
                    tmp_semantic['score'] = ref_semantic['score'] # no object with the attribute
            else: # and/or
                score_pool = []
                for ref_idx in semantic_info[qid][i][-1]:
                    tmp_semantic['dependency'].append(ref_idx)
                    ref_semantic = semantic_prob[ref_idx]
                    tmp_semantic['candidate'].update(ref_semantic['candidate'])
                    score_pool.append(ref_semantic['score'])
                if cur_op == 'or':
                    tmp_semantic['score'] = np.max(score_pool)
                else:
                    tmp_semantic['score'] = np.mean(score_pool)
            semantic_prob.append(tmp_semantic)

        # process the score into desired format
        tmp_score = []
        tmp_order = []
        for i in range(len(semantic_prob)):
            tmp_op = semantic_info[qid][i][0]
            if len(semantic_info[qid][i][-1]) == 0:
                tmp_order.append(1)
            else:
                tmp_order.append(tmp_order[semantic_info[qid][i][-1][0]]+1)
            tmp_score.append((tmp_op,tmp_order[i],str(round(semantic_prob[i]['score'],3))))
        final_result[qid] = tmp_score

    with open(os.path.join(args.save,'att_score.json'),'w') as f:
        json.dump(final_result,f)

main()
