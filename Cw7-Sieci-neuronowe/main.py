# Create, train and test neural network
# author: Radosław Pietkun

import idx2numpy
import numpy as np
from perceptron import create_net, classify_sample, train_net, activation_fun, test_net, train_net_with_ncross
from math import sqrt


# load data
train_images = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")  # 60000 x 28 x 28, [0-255]
train_labels = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")  # 60000 x 1 , [0-9]
test_images = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte")  # 10000 x 28 x 28, [0-255]
test_labels = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")  # 10000 x 1, [0-9]

# parameters
n_cross = 5  # n-cross validation param
bt = 0.001  # step in gradient method (how fast the net learns)
layers = [5]  # number of neurons in each layer (output layer excluded)


classes = np.unique(train_labels)  # classes
num_classes = classes.shape[0]
x_len = train_images.shape[1]*train_images.shape[2]  # number of input elements for net (28 x 28)

# create neural net
print("Creating net...")
weights = create_net(x_len, num_classes, layers)
print("Net created")

# prepare data: scale input data to (-1, 1)
train_images2 = np.copy(train_images)
train_images2 = train_images2 /255*2
train_images2 = train_images2  - 1

test_images2 = np.copy(test_images)
test_images2 = test_images2 /255*2
test_images2 = test_images2  - 1


# train neural network with n-cross validation
print("\nTrainind net...")
trained_weights = train_net_with_ncross(train_images2, train_labels, weights, n_cross, bt)
print("Net trained")


# test net using testing set
print("\nTesting net...")
success_count, num_samples = test_net(test_images2, test_labels, trained_weights)
accuracy = success_count / num_samples * 100  # [%]
print("Net tested")
print("Num_success, num_samples:", (success_count, num_samples))
print("Model accuracy on test images: {:.2f}%".format(accuracy))


# save weights to files
for i in range(len(trained_weights)):
    file1 = "w_" + str(i) + ".txt"
    np.savetxt(file1, trained_weights[i])


''' 
# load weights from files
w0 = np.loadtxt("w_0.txt")
w1 = np.loadtxt("w_1.txt")
#w2 = np.loadtxt("w_2.txt")
w = {}
w[0] = w0
w[1] = w1
'''
