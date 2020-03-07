import numpy as np
import json
import operator
import os
import argparse


parser = argparse.ArgumentParser(description="Obatining dictionary for questions and answers")
parser.add_argument("--question", type=str, required=True, help="path to GQA question")
args = parser.parse_args()

nb_answer = 1500
word_threshold = 3
questions = json.load(open(os.path.join(args.question,'train_balanced_questions.json')))
save_dir = './processed_data'

# select top answer
answer_bank=dict()
for qid in questions.keys():
	cur_ans = questions[qid]['answer']
	if cur_ans not in answer_bank:
		answer_bank[cur_ans] = 1
	else:
		answer_bank[cur_ans] += 1

total = np.sum(list(answer_bank.values()))
answer_bank = sorted(answer_bank.items(), key=operator.itemgetter(1)) #sorting the answers by frequency
answer_bank.reverse()
top_answer = dict()
count = 0
for i,ans in enumerate(answer_bank):
	if i >= nb_answer:
		break
	top_answer[ans[0]]=i
	count += ans[1]

with open(os.path.join(save_dir,'ans2idx_1500.json'),'w') as f:
	json.dump(top_answer,f)

print('Selected %d out of %d answers' %(len(top_answer),len(answer_bank)))
print('Number of valid samples is %d out of %d (%f percent)' %(count,total,count*100/total))

# create a word2idx mapping for questions
word_bank = dict()
for qid in questions.keys():
	cur_ans = questions[qid]['answer']
	if cur_ans not in top_answer:
		continue
	cur_question = questions[qid]['question']
	cur_question = str(cur_question).replace('?','').replace('.','').replace(',',' ')
	cur_question = cur_question.split(' ')
	for cur_word in cur_question:
		if cur_word not in word_bank:
			word_bank[cur_word] = 1
		else:
			word_bank[cur_word] += 1

word_bank = sorted(word_bank.items(), key=operator.itemgetter(1)) #sorting the answers by frequency
word_bank.reverse()
word2idx = dict()
for i,word in enumerate(word_bank):
	if word[1] >= word_threshold:
		word2idx[word[0]] = i+1
	else:
		break
word2idx['UNK'] = 0
with open(os.path.join(save_dir,'word2idx_1500.json'),'w') as f:
	json.dump(word2idx,f)

print('Number of selected vocabularies: %d'%len(word2idx))
