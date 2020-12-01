from keras.models import load_model
import cv2
import numpy as np
from scipy.spatial.distance import hamming
from scipy.spatial.distance import euclidean
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

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

def gen_confusion_matrix(preds, labels):
	classes = set(labels)
	n_classes = len(classes)

	# initialize confusion matrix
	conf_mat = np.zeros((n_classes, n_classes))

	# create map from classes to indices
	cl_map = {}
	i = 0
	for cl in classes:
		cl_map.update({cl: i})
		i = i + 1

	for index, label in enumerate(labels):
		lbl_map = cl_map[label]

		pred_cl = preds[index]
		pred_map = cl_map[pred_cl]

		conf_mat[lbl_map][pred_map] += 1

	return (conf_mat, cl_map)

# def plot_confusion_matrix():

def empirical_probs():
	class_map = get_class_map()

	train_classes = []
	with open('trainclasses.txt', 'r') as f:
		for cl in f:
			train_classes.append(cl.strip())

	ans = np.zeros(85)
	for cl in train_classes:
		ans += class_map[cl]

	ans = ans / len(train_classes)
	return np.array(ans)


def get_test_accuracy_bayesian(model):
	test_classes = []
	with open('testclasses.txt', 'r') as f:
		for line in f:
			test_classes.append(line.split()[0])

	class_map = get_class_map()

	img_dir = './JPEGImages_128x128/'

	pred_class = []
	labels = []

	emp_probs = empirical_probs()

	with open('test_images.txt', 'r') as f:
		for line in f:
			print(line)
			split = line.split()
			img_path = img_dir + split[0]
			label = split[1]

			img = cv2.imread(img_path)
			img = np.expand_dims(img, axis=0)
			pred = model.predict(img)[0]

			max_class = ""
			max_prob = 0

			for cl in test_classes:
				az = np.array(class_map[cl])

				# compute p(az | x)
				flip = 1 - az
				diff = np.abs(flip - pred)
				pazx = np.prod(diff)

				# compute p(az)
				pazi = np.abs(flip - emp_probs)
				paz = np.prod(pazi)

				pzx = pazx / paz

				if(pzx > max_prob):
					max_prob = pzx
					max_class = cl

			pred_class.append(max_class)
			labels.append(label)

	return (pred_class, labels)



def get_test_accuracy_euclidean(model):
	test_classes = []
	with open('testclasses.txt', 'r') as f:
		for line in f:
			test_classes.append(line.split()[0])

	class_map = get_class_map()

	img_dir = './JPEGImages_128x128/'

	pred_class = []
	labels = []

	attr_class = np.zeros(85)

	with open('test_images.txt', 'r') as f:
		for line in f:
			print(line)
			split = line.split()
			img_path = img_dir + split[0]
			label = split[1]

			img = cv2.imread(img_path)
			img = np.expand_dims(img, axis=0)
			pred = model.predict(img)[0]

			# get accuracy of attribute classifier
			attr_lab = np.array(class_map[label])
			attr_pred = np.round(pred)

			attr_class = attr_class + (attr_lab == attr_pred)

			min_class = ""
			min_dist = 1000000000
			for cl in test_classes:
				dist = euclidean(pred, class_map[cl])
				if(dist < min_dist):
					min_dist = dist
					min_class = cl

			pred_class.append(min_class)
			labels.append(label)

	return (pred_class, labels, attr_class)


def get_test_accuracy_hamming(model):
	test_classes = []
	with open('testclasses.txt', 'r') as f:
		for line in f:
			test_classes.append(line.split()[0])

	class_map = get_class_map()

	img_dir = './JPEGImages_128x128/'

	pred_class = []
	labels = []

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

			pred_class.append(min_class)
			labels.append(label)

	return (pred_class, labels)

# model = load_model('attribute_lenet.h5')
# (preds, labels, attr_acc) = get_test_accuracy_euclidean(model)
# acc = np.sum(preds == labels) / len(preds)
# print("Accuracy: " + str(acc))
# print(attr_acc)
# print(attr_acc / len(preds))

# np.save('preds_bayes.npy', preds)
# np.save('labels_bayes.npy', labels)

# preds = np.load('preds_bayes.npy')
# labels = np.load('labels_bayes.npy')
# print(preds)

# acc = np.sum(preds == labels) / len(preds)
# print(acc)

# (conf_mat, cl_map) = gen_confusion_matrix(preds, labels)

# conf_mat = np.load('conf_mat.npy')
# cl_map = np.load('cl_map.npy')

# # print(conf_mat)
# # conf_mat= conf_mat / conf_mat.sum(axis=1, keepdims=True)
# # print(conf_mat)


# keys = ['raccoon', 'pig', 'rat', 'chimp', 'seal', 'p+cat', 'h+whale', 'hippo', 'g+panda', 'leopard']
# ticks = range(10)
# xticks = [i + 0.5 for i in ticks]

# sns.heatmap(conf_mat, annot=True, fmt='g')
# plt.xticks(xticks, keys, fontsize=8)
# plt.yticks(xticks, keys, fontsize=8)
# plt.xlabel('Predicted')
# plt.ylabel('True label')
# plt.show()

predicates = []
with open('predicates.txt', 'r') as f:
	for line in f:
		predicates.append(line.split()[1])


acc = [0.53743737, 0.61188261, 0.86914817, 0.53672155, 0.66442377, 0.91252684,
 0.9964209,  0.93514674, 0.58296349, 0.44581246, 0.90379384, 0.73657838,
 0.74345025, 0.77924123, 0.68962062, 0.58697208, 0.54659986, 0.55347173,
 0.87158196, 0.94115963, 0.86356478, 0.70465283, 0.73528991, 0.82032928,
 0.97852541, 0.64538296, 0.66485326, 0.5115247,  0.6226199,  0.8539728,
 0.94745884, 0.6904796,  0.91438797, 0.64481031, 0.99699356, 0.95762348,
 0.85182534, 0.95146743, 0.82877595, 0.71381532, 0.61574803, 0.69663565,
 0.81488905, 0.70579814, 0.78425197, 0.87129563, 0.53657838, 0.64123121,
 0.71166786, 0.7567645,  0.65583393, 0.62004295, 0.62963493, 0.95375805,
 0.61918397, 0.85110952, 0.62290623, 0.69792412, 0.57451682, 0.82891911,
 0.95060845, 0.80128848, 0.83779528, 0.94430923, 0.82562634, 0.8210451,
 0.99885469, 0.65640659, 0.81259843, 0.63922691, 0.76077309, 0.63865426,
 0.73357194, 0.87372942, 0.82949177, 0.8519685,  0.67773801, 0.91567645,
 0.5758053,  0.6503937, 0.70350752, 0.56993558, 0.66098783, 0.52712956,
 0.52827487]

sorted_acc = sorted(acc)

cl_map = get_class_map()

seal_class = cl_map['seal']
hippo_class = cl_map['hippopotamus']

for i in range(85):
	if(seal_class[i] == hippo_class[i]):
		print(predicates[i])

# pred_accs_top10 = []
# pred_accs_worst10 = []

# for i, pred in enumerate(predicates):
# 	if acc[i] in sorted_acc[0:20]:
# 		pred_accs_worst10.append([pred, acc[i]])

# 	elif acc[i] in sorted_acc[-20:]:
# 		pred_accs_top10.append([pred, acc[i]])

# pred_accs_top10.sort(key=lambda x: x[1], reverse=True)
# pred_accs_worst10.sort(key=lambda x: x[1])
# print(tabulate(pred_accs_top10,tablefmt="latex", floatfmt=".2f"))
# print(tabulate(pred_accs_worst10, tablefmt="latex", floatfmt=".2f"))


