import numpy as np
import json
import pickle

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.300d.txt', encoding='utf-8')

lines = f.readlines()
print(len(lines))
for line in f:
	values = line.split()
	word = values[0]
	print(word.encode('utf-8'))

cnt = 0
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
	cnt = cnt + 1
	if(cnt % 1000 == 0): print(cnt)
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# print("Loading embeddings...")
# embeddings_index = np.load('glove_words.npy', allow_pickle=True)

print("Loading tokenizer...")
with open('tokenizer_qs.pickle', 'rb') as handle:
    qtokenizer = pickle.load(handle)

vocab_size = len(qtokenizer.word_index)+1
print(vocab_size)

print("Creating embedding matrix...")
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in qtokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

np.save('embedding_matrix.npy', embedding_matrix)

# e_mat = np.load('embedding_matrix.npy')
# for row in range(e_mat.shape[0]):
# 	if(np.count_nonzero(row) == 0):
# 		print(row)