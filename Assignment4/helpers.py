import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

def fit_question_tokenizer():
	with open('./train/v2_OpenEnded_mscoco_train2014_questions.json', encoding='utf-8') as json_data:
		train_qs = json.load(json_data)

	questions = []

	for q in train_qs['questions']:
		questions.append(q['question'])

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(questions)

	with open('tokenizer_qs.pickle', 'wb') as handle:
	    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fit_answer_tokenizer(num_words=1001):
	with open('./train/v2_mscoco_train2014_annotations.json', encoding='utf-8') as json_data:
		train_ans = json.load(json_data)

	answers = train_ans['annotations']

	ans = []
	for a in answers:
		ans_text = a['multiple_choice_answer']
		ans_text = ans_text.replace(' ', '')
		ans.append(ans_text)

	print("Fitting tokenizer...")
	tokenizer = Tokenizer(num_words)
	tokenizer.fit_on_texts(ans)

	with open('tokenizer_ans.pickle', 'wb') as handle:
	    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_answer_dictionary():
	answer_space = {}

	with open('./train/v2_mscoco_train2014_annotations.json', encoding='utf-8') as json_data:
		train_ans = json.load(json_data)

	answers = train_ans['annotations']

	for a in answers:
		ans_text = a['multiple_choice_answer']
		ans_new = ans_text.replace(' ', '')
		answer_space.update({ans_new: ans_text})

	return answer_space


