import numpy as np
import json
import pickle

preds = np.load('preds_10.npy')

with open('tokenizer_ans.pickle', 'rb') as handle:
    atokenizer = pickle.load(handle)

with open('answer_space.json') as json_data:
	answer_space = json.load(json_data)

with open('./val/v2_OpenEnded_mscoco_val2014_questions.json', encoding='utf-8') as json_data:
	val_qs = json.load(json_data)

questions = val_qs['questions']

json_data = []

inv_map = {v: k for k, v in atokenizer.word_index.items()}


for i, ans in enumerate(preds):
	q = questions[i]
	ans_pred = answer_space[inv_map[ans]]
	entry = {"answer": ans_pred, "question_id": q['question_id']}
	json_data.append(entry)

with open('v2_OpenEnded_mscoco_val2014_real_results.json', 'w') as f:
	json.dump(json_data, f)

