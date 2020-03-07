import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import os
import argparse

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['img_id', 'img_h','img_w','objects_id','objects_conf','attrs_id','attrs_conf','num_boxes', 'boxes', 'features']

parser = argparse.ArgumentParser(description="Extracting bottom-up features")
parser.add_argument("--input", type=str, required=True, help="path to bottom-up features")
parser.add_argument("--output", type=str, required=True, help="path to saving the extracted features")
args = parser.parse_args()

if __name__ == '__main__':
    os.mkdir(args.output,'features')
    os.mkdir(args.output,'box')
    
    with open(args.input) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            img_id = item['img_id']
            num_box = int(item['num_boxes'])
            cur_data = np.frombuffer(base64.b64decode(item['features']),
                  dtype=np.float32).reshape((num_box,-1))
            cur_box = np.frombuffer(base64.b64decode(item['boxes']),
                  dtype=np.float32).reshape((num_box,-1))
            np.save(os.path.join(args.output,'feature',str(img_id)),cur_data)
            np.save(os.path.join(args.output,'box',str(img_id)),cur_box)
