# Naive Bayes classifier
# author: Radosław Pietkun


import numpy as np
from statistics import mean, variance
from math import sqrt, pi, exp


def gauss(x, mean, variance):
    return 1/sqrt(2*pi*variance) * exp( -(x-mean)**2 / (2*variance) )


def calculate_gauss_params(samples):
    first_column = samples[:,0]
    class_id, counts = np.unique(first_column, return_counts=True)
    num_attributes = samples.shape[1]-1
    num_classes = len(class_id)
    gauss_params = np.zeros([num_classes, num_attributes*2+1])  # mean and variance for each class and each attribute
    gauss_params[:,0] = class_id
    for i in range(num_classes):
        for j in range(0,2*num_attributes,2):
            # find where 'i' class begins and ends in dataset 'samples'
            iter = i
            first_sample = 0  # id of first sample in class 'i'
            last_sample = counts[iter] - 1  # id of last sample in class 'i'
            while iter > 0:
                iter -= 1
                first_sample += counts[iter]
                last_sample += counts[iter]
            gauss_params[i, j+1] = mean(samples[first_sample:last_sample+1,int(j/2)+1])
            gauss_params[i, j+2] = variance(samples[first_sample:last_sample+1,int(j/2)+1])
    return gauss_params


# Build model based on 'samples' and classify 'testing_sample' to one of available classes
def bayes_classifier(samples, testing_sample):
    first_column = samples[:, 0]
    class_id, index, inverse, counts = np.unique(first_column, return_index=True, return_inverse=True, return_counts=True)
    classes = dict(zip(class_id, counts))
    num_samples, num_cols = samples.shape
    num_attributes = num_cols - 1

    Pc = np.zeros(len(classes))  # a priori probabilities for each class
    for i in range(len(classes)):
        Pc[i] = counts[i] / num_samples

    gauss_params = calculate_gauss_params(samples)

    Pcx = np.zeros(len(classes))  # a posterior probabilities for given sample and each class (denominator is skipped)
    for i in range(len(classes)):
        Pxc = 1  # conditional probabilities
        for j in range(0,2*num_attributes,2):
            Pxc *= gauss(testing_sample[int(j/2)], gauss_params[i, j+1], gauss_params[i, j+2])
        Pcx[i] = Pxc * Pc[i]

    id  = np.argmax(Pcx)
    classified_class = class_id[int(id)]
    return classified_class
