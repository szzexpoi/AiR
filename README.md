# AiR: Attention with Reasoning Capability
This code implements the Attention with Reasoning capability (AiR) framework. It contains three principal components"
- AiR-E: an quantitative evaluation method for measuring the alignments between different attentions and reasoning process,
- AiR-M: an attention supervision method for learning attention progressively throughout the reasoning process, and
- AiR-D: the first human eye-tracking dataset for Visual Question Answering task

### reference
If you use our code or data, please cite our paper:
```
Anonymous submission for ECCV 2020, paper ID 445.
```

### Disclaimer
We adopt the official implementation of the [Bilinear Attention Network](https://github.com/jnhwkim/ban-vqa) as a backbone model for attention supervision. We use the bottom-up features from [LXMERT](https://github.com/airsplay/lxmert). Please refer to these links for further README information.   

### Requirements
1. Requirements for Pytorch. We used Pytorch 1.2.0 in our experiments.
2. Python 3.6+
3. Python 2.7
4. You may need to install the OpenCV package (CV2) for Python.

### Data Pre-processing
1. Download the [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).
2. Download the [bottom-up features](https://github.com/airsplay/lxmert) and unzip it.
3. Extracting features from the raw tsv files (**Important**: You need to run the code with Python 2):
  ```
  python2 ./pre_processing/extract_tsv.py --input $TSV_FILE --output $FEATURE_DIR
  ```
4. Generate our atomic operations abstracted from GQA annotations:
  ```
  python ./pre_processing/process_semantics.py --question $GQA_ROOT/question --scene_graph $GQA_ROOT/scene_graph --mapping ./data --save ./data
  ```
5. Generate ground truth attention mask for supervision:
  ```
  python ./pre_processing/generate_att_mask.py --bbox_dir $FEATURE_DIR/box --question $GQA_ROOT/question --scene_graph $GQA_ROOT/scene_graph --semantics ./data --save ./AiR-M/processed_data
  ```

### Evaluating Attention (AiR-E)
The AiR-E score can be computed with:
  ```
  python ./AiR-E/attention_eval.py --image $GQA_ROOT/images --question $GQA_ROOT/question --scene_graph $GQA_ROOT/scene_graph --semantics ./data --att_dir $ATTENTION_DIR --bbox_dir $FEATURE_DIR/box --att_type ATTENTION_TYPE --save ./data
  ```

For evaluating the human attention in our dataset, specify **ATTENTION_TYPE** as `human` and **$ATTENTION_DIR** as the directory storing the saliency maps. For evaluating machine attention, specify **ATTENTION_TYPE** as `spatial` or `object` and **$ATTENTION_DIR** as the file storing the attention. Our code assumes the machine attention is stored in a Numpy dictionary where the keys are qid and values are the maps. We use spatial attention with spatial size 7x7 and object-based attention with size 36.

The output is a Json file storing the scores. You can access the scores for a specific question (for example qid `011000868`) as:
  ```
  >>> import json
  >>> data = json.load(open('./data/att_score.json'))
  >>> data['011000868']
  [['select', 1, '7.908'], ['relate', 2, '7.053'], ['query', 3, '6.197']]
  ```
For each step, the result contains the reasoning operation, level of dependency in the process, and the AiR-E score.

### Human Eye-tracking dataset for VQA (AiR-D)
Our data is available at https://drive.google.com/file/d/1tMLJtgaj4Bh-QSfKDBEJKnokN9Il7BQB/view?usp=sharing.
