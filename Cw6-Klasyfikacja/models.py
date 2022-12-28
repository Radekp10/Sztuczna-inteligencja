# Functions classify_sample(), test_model(), find_best_model()
# author: Radosław Pietkun


from id3 import id3
import numpy as np


def classify_sample(root, testing_sample):
    if not root.has_children():
        #if root.value is None:  # can be uncommented for diagnostic purposes
        #    print("Unable to classify sample {}. There were no enough training "
        #          "samples during building tree to set class in this leaf".format(testing_sample))
        return root.value  # class_id in leaf
    attr_id = root.value  # attribute_id in inner node
    attr_val = testing_sample[attr_id]  # attribute value
    return classify_sample(root.children[attr_val], testing_sample)


def test_model(root, U_testing, class_column_id):
    missclass_count = 0  # count of misclassified samples
    classify_failures = 0  # count of classification failures (model was unable to classify sample)
    num_samples = U_testing.shape[0]
    for i in range(num_samples):
        classified_class = classify_sample(root, U_testing[i])
        if classified_class is None:  # unable to classify sample
            classify_failures += 1
            continue
        if U_testing[i, class_column_id] != classified_class:  # misclassification
            missclass_count += 1
    return missclass_count, classify_failures, num_samples


def find_best_model(U, n_cross, D, class_column_id, max_depth):
    num_samples = U.shape[0]
    model_accuracy = np.zeros(n_cross)
    models = []  # there will be 'n' models, model is represented by a root node
    ranges = []  # list of 'n' ranges, each range contains about 1/n indices of 'U' samples
    ranges.append( range(0, int(num_samples/n_cross)))  # first range
    for i in range(1, n_cross-1):
        ranges.append( range(i*int(num_samples/n_cross), (i+1)*int(num_samples/n_cross)) )
    ranges.append( range((n_cross-1)*int(num_samples/n_cross), num_samples) )  # last range

    for i in range(n_cross):
        U_validating = U[ranges[i], :]  # i-th subset will be used to validate model
        U_training = np.zeros([0, U.shape[1]])  # all subsets except i-th will be used to train model
        for j in range(n_cross):
            if j != i:
                U_training = np.concatenate((U_training, U[ranges[j], :]))

        models.append( id3(D, U_training, class_column_id, max_depth) )  # build i-th model
        missclass_count, classify_failures, num_test_samples = test_model(models[i], U_validating, class_column_id)  # validate i-th model
        model_accuracy[i] = (1-(missclass_count+classify_failures)/num_test_samples)*100  # [%]

    best_model_id = np.argmax(model_accuracy)  # find best model
    print("Models accuracies (based on validation set): ", model_accuracy)
    print("Best model with id: ", best_model_id)
    return models[best_model_id]
