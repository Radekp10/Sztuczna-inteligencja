# Perceptron implementation
# author: Radosław Pietkun

import numpy as np
import math


# activation function
def activation_fun(x):
    return np.exp(x) / (1 + np.exp(x))  # sigmoid function


# activation function derivative
def activation_fun_deriv(x):
    return activation_fun(x) * (1-activation_fun(x))


# x_len - number of elements in input vector
# num_classes - number of classes (number of outputs of net)
# layers - list with numbers of neurons in each layer except the last one (output layer excluded)
def create_net(x_len, num_classes, layers):
    num_layers = len(layers)

    # Sequence: x -> y0 -> w0 -> y1 -> w1 -> ... -> y[num_layers+1]
    w = {}  # dict for weights: key - layer id, value - 2D array (num of neurons in layer * len of input vector)
    y = {}  # dict for inputs: key - layer id, value - 1D vector, input for neurons in 'key' layer

    # input vector for first neuron layer
    y[0] = np.zeros(x_len+1)

    for j in range(num_layers):  # for each layer except the last one
        ran_uni_min = -1/math.sqrt(len(y[j]))  # min for random uniform distribution
        ran_uni_max = 1/math.sqrt(len(y[j]))   # max for random uniform distribution
        w[j] = np.random.uniform(ran_uni_min, ran_uni_max, (layers[j], y[j].shape[0]))  # weights in j-th layer
        y[j+1] = np.zeros(layers[j]+1)  # output of j-th layer

    w[num_layers] = np.zeros((num_classes, y[num_layers].shape[0]))  # weights in output layer
    y[num_layers+1] = np.zeros(num_classes)  # output vector

    # return dict of net's weights
    return w


# w -dict with weights
# bt - step, how fast the model learns
def train_net(train_images, train_labels, w, bt):
    num_samples = train_images.shape[0]
    num_layers = len(w) - 1  # all layers except the last one
    num_classes = w[num_layers].shape[0]  # number of classes (number of neurons in last layer)
    x_len = train_images.shape[1]*train_images.shape[2]  # number of elements (values) in single sample

    for s in range(num_samples):  # for each training sample

        #if s % 1000 == 0:  # diagnistic
        #    print("training... current sample no.:", s)

        # Sequence: x -> y0 -> a0=w0*y0, act_fun(a0)=y1 -> y1 -> a1=w1*y1, ... -> ... -> y[num_layers+1]
        y = {}  # dict for inputs: key - layer id, value - 1D vector, input for neurons in 'key' layer and also output of neurons in 'key-1' layer
        a = {}  # dict for scalar products: key - layer id, value - 1D vector, scalar products in 'key' layer
        error = {}  # dict for errors: key - layer id (connected with 'y' dict), value - 1D vector, errors for each neuron in 'key' layer

        # I. Go with train sample through the net forward and calculate error for each class

        # input vector for first neuron layer
        y[0] = np.zeros(x_len + 1)
        y[0][:x_len] = train_images[s].flatten()  # 2D -> 1D
        y[0][-1] = 1  # additional input (always 1)

        for j in range(num_layers):  # for each layer except the last one
            y[j + 1] = np.zeros(w[j].shape[0] + 1)  # output of j-th layer
            a[j] = np.zeros(w[j].shape[0])  # scalar products for each neuron in j-th layer
            for i in range(w[j].shape[0]):  # for each neuron in j-th layer
                a[j][i] = sum(np.multiply(w[j][i], y[j]))  # j-th layer, i-th neuron
                y[j + 1][i] = activation_fun(a[j][i])
            y[j + 1][-1] = 1  # additional input=1 in each layer except the output layer

        # last layer (output layer)
        y[num_layers + 1] = np.zeros(num_classes)  # output vector
        a[num_layers] = np.zeros(num_classes)
        error[num_layers + 1] = np.zeros(num_classes)
        for i in range(num_classes):
            a[num_layers][i] = sum(np.multiply(w[num_layers][i], y[num_layers]))  # inside neurons in output layer
            y[num_layers + 1][i] = a[num_layers][i]  # linear neurons (no activation function)
            if i == train_labels[s]:  # 1 output should be '1' (for class match)
                error[num_layers + 1][i] = (y[num_layers + 1][i] - 1)  # error between calculated and desired output
            else:  # other outputs should be '-1'
                error[num_layers + 1][i] = (y[num_layers + 1][i] - -1)  # error between calculated and desired output

        # II. Propagate errors backwards
        for j in range(num_layers,0,-1):  # for each layer
            error[j] = np.zeros(w[j-1].shape[0])
            for i in range(w[j-1].shape[0]):  # for each neuron
                error[j][i] = sum(np.multiply(w[j][:,i], error[j+1]))

        # III. Modify weights
        for j in range(num_layers):  # for each layer (except output layer)
            for i in range(w[j].shape[0]):  # for each neuron
                w[j][i] = w[j][i] - bt * error[j+1][i] * activation_fun_deriv(a[j][i]) * y[j]
                # vector = vector - scalar * scaler     *      scalar                  *  vector
        # output layer (linear neurons -> derivative=1)
        for i in range(w[num_layers].shape[0]):  # for each neuron in output layer (for each class)
            w[num_layers][i] = w[num_layers][i] - bt * error[num_layers+1][i] * y[num_layers]

    # return modified weights
    return w


# w -dict with weights
# n_cross - n-cross validation param
# bt - step, how fast the model learns
def train_net_with_ncross(train_images, train_labels, w, n_cross, bt):
    num_samples = train_images.shape[0]
    model_accuracy = np.zeros(n_cross)
    models = []  # there will be 'n' models, model is represented by weights dict
    ranges = []  # list of 'n' ranges, each range contains about 1/n indices of 'train_images' samples
    ranges.append( range(0, int(num_samples/n_cross)))  # first range
    for i in range(1, n_cross-1):
        ranges.append( range(i*int(num_samples/n_cross), (i+1)*int(num_samples/n_cross)) )
    ranges.append( range((n_cross-1)*int(num_samples/n_cross), num_samples) )  # last range

    for i in range(n_cross):
        print("N-cross validation progress: {}/{}:".format(i+1, n_cross))
        validating_set = train_images[ranges[i], :, :]  # i-th subset will be used to validate model
        validatin_set_labels = train_labels[ranges[i]]
        training_set = np.zeros([0, train_images.shape[1], train_images.shape[2]])  # all subsets except i-th will be used to train model
        training_set_labels = np.zeros(0)
        for j in range(n_cross):
            if j != i:
                training_set = np.concatenate((training_set, train_images[ranges[j], :, :]))
                training_set_labels = np.concatenate((training_set_labels, train_labels[ranges[j]]))

        w_trained = train_net(training_set, training_set_labels, w, bt)  # train i-th model
        models.append(w_trained)
        success_count, num_samples = test_net(validating_set, validatin_set_labels, w_trained)  # validate i-th model
        model_accuracy[i] = (success_count/num_samples)*100  # [%]

    best_model_id = np.argmax(model_accuracy)  # find best net
    print("Models accuracies (based on validation set) [%]: ", model_accuracy)
    print("Best model with id: ", best_model_id)
    return models[best_model_id]


# w - net's weights (dictionary of matrices)
def test_net(test_images, test_labels, w):
    num_samples = test_images.shape[0]
    sample_len = test_images.shape[1]*test_images.shape[2]  # number of elements (values) in single sample
    success_count = 0  # classification successes
    for i in range(num_samples):
        #if i % 1000 == 0:  # diagnostic
        #    print("testing... current sample no.:", i)
        sample = test_images[i].flatten()  # 2D -> 1D
        classified_class = classify_sample(sample, w)
        if classified_class == test_labels[i]:
            success_count += 1
    return success_count, num_samples


# sample - sample to classify (should be 1D vector)
# w - dict of net's weights
def classify_sample(sample, w):
    if len(sample.shape) > 1:  # convert to 1D if 'sample' is 2D
        sample = sample.flatten()

    num_layers = len(w) - 1  # all layers except the last one
    num_classes = w[num_layers].shape[0]  # number of classes (number of neurons in last layer)

    # Sequence: x -> y0 -> w0 -> y1 -> w1 -> ... -> y[num_layers+1]
    y = {}  # dict for inputs: key - layer id, value - 1D vector, input for neurons in 'key' layer

    # input vector for first neuron layer
    y[0] = np.zeros(len(sample)+1)
    y[0][:len(sample)] = sample
    y[0][-1] = 1

    for j in range(num_layers):  # for each layer except the last one
        y[j+1] = np.zeros(w[j].shape[0]+1)  # output of j-th layer
        for i in range(w[j].shape[0]):  # for each neuron in j-th layer
            y[j+1][i] = activation_fun(sum(np.multiply(w[j][i], y[j])))
        y[j+1][-1] = 1

    y[num_layers+1] = np.zeros(num_classes)  # output vector
    for i in range(num_classes):
        y[num_layers+1][i] = sum(np.multiply(w[num_layers][i], y[num_layers]))

    best_matches = np.argwhere(y[num_layers+1] == np.amax(y[num_layers+1]))  # find all classes that match the best
    chosen_class = np.random.choice(best_matches.flatten())  # choose randomly one of them

    return chosen_class
