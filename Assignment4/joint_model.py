from keras.models import *
from keras.layers import *
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
import json
import pickle

def data_generator(questions, answers, images, tokenizer_qs, tokenizer_ans, batch_size, max_len):

	i = 0
	while True:
		q_batch = []
		a_batch = []
		img_batch = []

		for b in range(batch_size):
			# reset to beginning of batch
			if i == len(questions):
				i = 0

			q = questions[i]
			a = answers[i]

			q_vec = tokenizer_qs.texts_to_sequences([q['question']])[0]

			# since tokenizer has 1001 words, take the last 1000 elements (prevent off-by-one error)
			ans_text = a['multiple_choice_answer'].replace(' ', '')
			a_vec = tokenizer_ans.texts_to_matrix([ans_text])[0][1:]
			
			img_vec = images[q['image_id']]

			if(np.count_nonzero(a_vec) != 0):
				q_batch.append(q_vec)
				a_batch.append(a_vec)
				img_batch.append(img_vec)

			i += 1

		q_batch = pad_sequences(q_batch, padding='post', maxlen=max_len)
		yield [np.array(q_batch), np.array(img_batch)], np.array(a_batch)


def VQA_model(num_words, embedding_matrix, n_classes=1000, max_len=30, img_len=2048):

        # LSTM for question input
        text_input = Input(shape=(max_len,))
        embed = Embedding(num_words, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)(text_input)
        lstm1 = LSTM(512, return_sequences=True)(embed)
        lstm2 = LSTM(512)(lstm1)

        # FC layer for ResNet embeddings
        img_input = Input(shape=(img_len,))
        fc1 = Dense(512, activation='relu')(img_input)

        # Merge the two layers by element-wise addition of text and image embeddings
        merge = Add()([lstm2, fc1])

        # Add FC layers and perform softmax
        dense1 = Dense(1000, activation='relu')(merge)
        dense2 = Dense(1000, activation='softmax')(dense1)

        model = Model([text_input, img_input], dense2)
        return model

print("Loading questions...")
with open('./train/v2_OpenEnded_mscoco_train2014_questions.json', encoding='utf-8') as json_data:
	train_qs = json.load(json_data)

print("Loading answers...")
with open('./train/v2_mscoco_train2014_annotations.json', encoding='utf-8') as json_data:
	train_ans = json.load(json_data)

questions = train_qs['questions']
answers = train_ans['annotations']

print("Loading images...")
images = pickle.load(open("./train/train.pickle", "rb"))

print("Loading tokenizers...")
with open('tokenizer_ans.pickle', 'rb') as handle:
    tokenizer_ans = pickle.load(handle)

with open('tokenizer_qs.pickle', 'rb') as handle:
    tokenizer_qs = pickle.load(handle)

e_mat = np.load('embedding_matrix.npy')

batch_size = 32
max_len = 30

generator = data_generator(questions, answers, images, tokenizer_qs, tokenizer_ans, batch_size, max_len)

print("Defining model...")
model = VQA_model(num_words=len(tokenizer_qs.word_index)+1, embedding_matrix=e_mat)
lr=0.001
momentum=0.9
optimizer = SGD(lr=lr, momentum=momentum)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

mc = ModelCheckpoint('modelab{epoch:08d}.h5', period=5)

callback = [mc]

print("Fitting model")
model.fit_generator(generator, 
	steps_per_epoch=len(questions)/batch_size,
	callbacks=callback, 
	epochs=20)

model.save('vqa_model.h5')


