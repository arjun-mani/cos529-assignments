import shutil
import os
import cv2
import numpy as np
import h5py

# get mapping between classes and attributes
class_order = []
with open('classes.txt', 'r') as f:
	for line in f:
		class_order.append(line.split()[1])

class_map = {}

index = 0
with open('predicate-matrix-binary.txt', 'r') as f:
	for line in f:
		attributes = line.split()
		attributes = [int(i) for i in attributes]

		class_map.update({class_order[index]: attributes})
		index = index + 1


# read in training classes
train_classes = []

with open('trainclasses.txt', 'r') as f:
	for cl in f:
		train_classes.append(cl.strip())

train_set = []
train_label = []

for cl in train_classes:
	print(cl)
	train_dir = './JPEGImages_128x128/' + cl

	for img_name in os.listdir(train_dir):
		img_path = train_dir + '/' + img_name
		img = cv2.imread(img_path)

		train_set.append(img)
		train_label.append(class_map[cl])

train_set = np.array(train_set)
h5f = h5py.File('trainset.h5', 'w')
h5f.create_dataset('dataset_1', data=train_set)
h5f.close()
# np.savez_compressed('train_images', a=train_set)

# with h5py.File('trainset.h5', 'r') as hf:
#     data = hf['dataset_1'][:]

# print(data.shape)

# train_label = np.array(train_label)
# np.save('train_labels.npy', train_label)
