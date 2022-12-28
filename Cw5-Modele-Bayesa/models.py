# Functions used to building and verifying models
# author: Radosław Pietkun


from bayes import calculate_gauss_params, gauss
import numpy as np
from statistics import mean


# Train model with 'training_samples', test model with "testing_samples',
# take into account only attributes in list 'attributes',
# return number of misslassifications for 'training_samples'
def try_model(training_samples, testing_samples, attributes):
    gauss_params = calculate_gauss_params(training_samples)  # for each class and for each attribute

    num_samples = training_samples.shape[0]
    num_missclass = 0

    first_column = training_samples[:,0]
    class_id, counts = np.unique(first_column, return_counts=True)

    Pc = np.zeros(len(class_id))  # a priori probabilities for each class
    for i in range(len(class_id)):
        Pc[i] = counts[i] / num_samples

    for i in range(len(testing_samples)):
        atrr = testing_samples[i,1:]  # attributes for one sample
        correct_class = int(testing_samples[i,0])  # the proper class for sample

        Pcx = np.zeros(len(class_id))  # a posterior probabilities for given sample and each class (denominator is skipped)
        for i in range(len(class_id)):
            Pxc = 1  # conditional probabilities
            for j in attributes:
                Pxc *= gauss(atrr[j], gauss_params[i, 2*j + 1], gauss_params[i, 2*j + 2])
            Pcx[i] = Pxc * Pc[i]

        id = np.argmax(Pcx)
        classified_class = class_id[int(id)]  # classify sample to class with the highest probability

        if classified_class != correct_class:
            num_missclass += 1

    return  num_missclass


# Split 'samples' into 'n' parts,
# return list with 'n' elements
def split_samples(samples, n):  # split one array into 'n' arrays and put them in the list
    samples_subsets = []
    num_samples = samples.shape[0]
    coef = num_samples/n  # average number of samples in each set
    first_sample = 0  # included
    for i in range(1,n+1):
        last_sample = int(i*coef)  # excluded
        samples_subsets.append(samples[first_sample:last_sample])
        first_sample = last_sample
    return samples_subsets


# Find best attribute for classifying 'samples' with n-cross-validation,
# return best attribute id and model quality
def find_best_attr(samples, n):
    first_column = samples[:, 0]
    class_id, counts = np.unique(first_column, return_counts=True)

    num_samples, num_cols = samples.shape
    num_attributes = num_cols - 1
    num_classes = len(class_id)

    samples_sets = np.zeros([num_classes, n]).tolist()  # each element of 2D list contains some samples (2D arrays)
    for i in range(num_classes):
        iter = i
        first_sample = 0  # id of first sample in class 'i'
        last_sample = counts[iter] - 1  # id of last sample in class 'i'
        while iter > 0:
            iter -= 1
            first_sample += counts[iter]
            last_sample += counts[iter]
        samples_sets[i] = split_samples(samples[first_sample:last_sample + 1], n)  # split samples from each class into 'n' subsets

    model_quality = np.zeros(num_attributes)
    for attr in range(num_attributes):
        num_missclass = np.zeros(n)
        for j in range(n):  # create 1 training set and 1 testing set from 'n' sets
            testing_samples = np.zeros([0, num_attributes + 1])
            training_samples = np.zeros([0, num_attributes + 1])
            for k in range(num_classes):
                for p in range(n):
                    if p != j:
                        training_samples = np.concatenate((training_samples, samples_sets[k][p]))
                    else:
                        testing_samples = np.concatenate((testing_samples, samples_sets[k][p]))
            num_missclass[j] = try_model(training_samples, testing_samples, [attr])

        model_quality[attr] = mean(num_missclass)

    best_attr_id = np.argmin(model_quality)
    return  best_attr_id, model_quality


# Build model with 'samples' using n-cross-validation,
# take into account only attributes in list 'attributes_to_be_tested',
# return model quality
# This function is quite similar to find_best_attr() but has different purpose
def test_model_with_chosen_attributes(samples, n, attributes_to_be_tested):
    first_column = samples[:, 0]
    class_id, counts = np.unique(first_column, return_counts=True)

    num_samples, num_cols = samples.shape
    num_attributes = num_cols-1
    num_classes = len(class_id)

    # Split samples into 'n' subarrays
    samples_sets = np.zeros([num_classes, n]).tolist()  # each element of 2D list contains some samples (2D arrays)
    for i in range(num_classes):
        iter = i
        first_sample = 0  # id of first sample in class 'i'
        last_sample = counts[iter] - 1  # id of last sample in class 'i'
        while iter > 0:
            iter -= 1
            first_sample += counts[iter]
            last_sample += counts[iter]
        samples_sets[i] = split_samples(samples[first_sample:last_sample + 1], n)

    num_missclass = np.zeros(n)
    for j in range(n):
        testing_samples = np.zeros([0, num_attributes + 1])
        training_samples = np.zeros([0, num_attributes + 1])
        for k in range(num_classes):
            for p in range(n):
                if p != j:
                    training_samples = np.concatenate((training_samples, samples_sets[k][p]))
                else:
                    testing_samples = np.concatenate((testing_samples, samples_sets[k][p]))
        num_missclass[j] = try_model(training_samples, testing_samples, attributes_to_be_tested)

    model_quality = mean(num_missclass)
    return model_quality
