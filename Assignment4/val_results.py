from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import pickle

print("Loading model...")
model = load_model('modelab00000010.h5')

print("Loading questions...")
with open('./val/v2_OpenEnded_mscoco_val2014_questions.json', encoding='utf-8') as json_data:
	val_qs = json.load(json_data)

questions = val_qs['questions']

print("Loading images...")
images = pickle.load(open("./val/val.pickle", "rb"))

print("Loading tokenizers...")

with open('tokenizer_ans.pickle', 'rb') as handle:
    atokenizer = pickle.load(handle)

with open('tokenizer_qs.pickle', 'rb') as handle:
    qtokenizer = pickle.load(handle)

with open('answer_space.json') as json_data:
	answer_space = json.load(json_data)


json_data = []
batch_size = 32
preds = []
for i in range(0, len(questions), batch_size):
	print(i)
	# if(i % 1000 == 0): print(i)
	q = questions[i:min(i+batch_size, len(questions))]

	q_batch = [x['question'] for x in q]
	img_vec = [images[x['image_id']] for x in q]

	q_vec = qtokenizer.texts_to_sequences(q_batch)

	q_vec = pad_sequences(q_vec, padding='post', maxlen=30)
	img_vec = np.array(img_vec)

	pred = model.predict(q_vec)
	ans = np.argmax(pred, axis=-1)+1
	preds.extend(ans)

preds = np.array(preds)
np.save('preds_10.npy', preds)


# with open('v2_OpenEnded_mscoco_val2014_real_results.json', 'w') as f:
# 	json.dump(json_data, f)
