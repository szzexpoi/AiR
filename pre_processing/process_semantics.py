import json
import numpy as np
import os
import operator
import re
import argparse

parser = argparse.ArgumentParser(description="Abstracting semantics from GQA annotations")
parser.add_argument("--question", type=str, required=True, help="path to GQA question")
parser.add_argument("--scene_graph", type=str, required=True, help="path to GQA scene graph")
parser.add_argument("--mapping", type=str, default='./data', help="path to semantic mapping")
parser.add_argument("--save", type=str, default='./data', help="path for saving the data")
args = parser.parse_args()

# search for dependent grounded objects
def search_dep(semantics,dep_idx):
    while len(semantics[dep_idx]['dependencies'])>0 and ((re.search(r'\(([^)]+)\)', semantics[dep_idx]['argument']) is None) or not (re.search(r'\(([^)]+)\)', semantics[dep_idx]['argument']).group(1).isdigit())):
        dep_idx = semantics[dep_idx]['dependencies'][0]
    if re.search(r'\(([^)]+)\)', semantics[dep_idx]['argument']) is None:
        return 'obj*:'+semantics[dep_idx]['argument']
    elif len(re.search(r'\(([^)]+)\)', semantics[dep_idx]['argument']).group(1))<=3:
        return 'obj*:'+semantics[dep_idx]['argument']
    else:
        return 'obj:'+re.search(r'\(([^)]+)\)', semantics[dep_idx]['argument']).group(1)

# abstracting the semantics for training/validation split of GQA
for split in ['train','val']:
    question = json.load(open(os.path.join(args.question,split+'_balanced_questions.json')))
    scene_graph = json.load(open(os.path.join(args.scene_graph,split+'_sceneGraphs.json')))
    simplified_mapping = json.load(open(os.path.join(args.mapping,'simplified_mapping.json')))

    processed_semantic = dict()
    for qid in question.keys():
        img_id = question[qid]['imageId']
        semantics = question[qid]['semantic']
        processed_semantic[qid] = []
        invalid_count = 0
        for i in range(len(semantics)):
            cur_semantic = semantics[i]
            if cur_semantic['operation'] == 'exist':
                invalid_count += 1
                continue # redundant operation
            simplified_structure = simplified_mapping[cur_semantic['operation']]
            cur_op = simplified_structure['operation']
            # processing the first argument independently
            if simplified_structure['argument_1'] == 'attr':
                cur_arg_1 = cur_semantic['argument']
                cur_arg_2 = search_dep(semantics,cur_semantic['dependencies'][0])
            elif simplified_structure['argument_1'] == 'obj':
                if (re.search(r'\(([^)]+)\)', cur_semantic['argument']) is not None) and ((re.search(r'\(([^)]+)\)', cur_semantic['argument']).group(1)).split(',')[0].isdigit()):
                    cur_arg_1 = 'obj:'+ re.search(r'\(([^)]+)\)', cur_semantic['argument']).group(1)
                else:
                    cur_arg_1 = 'obj*:' + cur_semantic['argument'].split(',')[0]
                if simplified_structure['argument_2'] == 'dep':
                    cur_arg_2 = search_dep(semantics,cur_semantic['dependencies'][0])
                else:
                    cur_arg_2 = None
            else: # the two arguments are entangled together
                if simplified_structure['argument_2'] == 'dep' and len(cur_semantic['dependencies'])==2:
                    cur_arg_1 = search_dep(semantics,cur_semantic['dependencies'][0])
                    cur_arg_2 = search_dep(semantics,cur_semantic['dependencies'][1])
                else:
                    cur_arg_1 = search_dep(semantics,cur_semantic['dependencies'][0])
                    cur_arg_2 = 'None'
            if len(cur_semantic['dependencies']) > 0:
                tmp_dep = [cur-invalid_count for cur in cur_semantic['dependencies']]
            else:
                tmp_dep = cur_semantic['dependencies']
            processed_semantic[qid].append((cur_op,cur_arg_1,cur_arg_2,tmp_dep))

    with open(os.path.join(args.save,'simplified_semantics_'+split+'.json'),'w') as f:
        json.dump(processed_semantic,f)
