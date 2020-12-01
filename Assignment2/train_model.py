from keras.layers import *
from keras.models import *
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import os
import cv2
import numpy as np
from scipy.spatial.distance import hamming

def attribute_model(img_width=128, img_height=128, n_attributes=85):
	img_input = Input(shape=(img_width, img_height, 3))

	# use VGG 16 architecture 
	# with an input size of 128x128, the output of the VGG conv layers will be 4x4x512
	vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=img_input)

	# Flatten into 8192-length vector
	x = Flatten()(vgg16.output)

	# add FC layers for classification
	x = Dense(4096, activation='relu')(x)
	x = Dense(4096, activation='relu')(x)
	x = Dense(n_attributes, activation='sigmoid')(x)

	model = Model(img_input, x)
	return model

# def attribute_model(img_width=128, img_height=128, n_attributes=85):
# 	img_input = Input(shape=(img_width, img_height, 3))

# 	x = Conv2D(20, (5, 5), padding='same', activation='relu')(img_input)
# 	x = MaxPooling2D((2, 2), strides=(2, 2))(x)

# 	x = Conv2D(40, (5, 5), padding='same', activation='relu')(x)
# 	x = MaxPooling2D((2, 2), strides=(2, 2))(x)

# 	x = Flatten()(x)

# 	x = Dense(4096, activation='relu')(x)
# 	x = Dense(n_attributes, activation='sigmoid')(x)

# 	model = Model(img_input, x)
# 	return model

def get_class_map():
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

	return class_map

def get_training_data():
	class_map = get_class_map()

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

	return (np.array(train_set), np.array(train_label))

def get_test_accuracy(model):
	test_classes = []
	with open('testclasses.txt', 'r') as f:
		for line in f:
			test_classes.append(line.split()[0])

	class_map = get_class_map()

	img_dir = './JPEGImages_128x128/'

	acc = 0
	cnt = 0

	with open('test_images.txt', 'r') as f:
		for line in f:
			print(line)
			split = line.split()
			img_path = img_dir + split[0]
			label = split[1]

			img = cv2.imread(img_path)
			img = np.expand_dims(img, axis=0)
			pred = model.predict(img)[0]
			pred = np.round(pred)

			min_class = ""
			min_dist = 1
			for cl in test_classes:
				dist = hamming(pred, class_map[cl])
				if(dist < min_dist):
					min_dist = dist
					min_class = cl

			if(min_class == label): 
				acc += 1

			cnt += 1

	return acc / cnt


print("Creating training data...") 
(X_train, Y_train) = get_training_data()

print("Fitting model...")
model = attribute_model()
lr = 0.001
mom = 0.9
opt = SGD(lr=lr, momentum=mom)
model.compile(loss='binary_crossentropy', optimizer=opt)

# X_train = np.load('train_images.npz')
# Y_train = np.load('train_labels.npy')

model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print("Saving model...")
model.save('attribute_lenet.h5')

print("Testing model...")

acc = get_test_accuracy(model)
print("Accuracy: " + str(acc))

