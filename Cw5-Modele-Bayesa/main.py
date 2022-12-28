# Testing script
# author: Radosław Pietkun


from bayes import bayes_classifier
from models import find_best_attr, test_model_with_chosen_attributes
import numpy as np


if __name__ == '__main__':
    file = "wine.data"
    n = 5  # n-cross validation parameter
    attributes = {
        1: "Alcohol",
        2: "Malic acid",
        3: "Ash",
        4: "Alcalinity of ash",
        5: "Magnesium",
        6: "Total phenols",
        7: "Flavanoids",
        8: "Nonflavanoid phenols",
        9: "Proanthocyanins",
        10: "Color intensity",
        11: "Hue",
        12: "OD280 / OD315 of diluted wines",
        13: "Proline"
    }

    samples = np.loadtxt(file, delimiter=",")  # it's assumed that data is sorted by class id

    # Find best attribute for classification purposes
    print("I. Finding best attribute...")
    best_attr, models_qualities = find_best_attr(samples, n)
    print("Average number od missclassifications for each attribute:\n", models_qualities)
    print("Best attribute for classification: ", attributes[best_attr+1])  # attributes are indexed from 1
    print("Best model quality: ", models_qualities[best_attr])

    # Check model quality for attributes: 7 (best found) and 10
    print("\nII. Testing model with 2 attributes: Flavanoids and Color intensity...")
    new_model_quality = test_model_with_chosen_attributes(samples, n, [best_attr, 10-1])
    # Adding further attributes to model:
    # new_model_quality = test_model_with_chosen_attributes(samples, n, [6, 9, 12, 0, 5, 10, 11, 1, 8, 4, 7, 3, 2])
    print("Model quality: ", new_model_quality)
    if new_model_quality < models_qualities[best_attr]:
        print("Result: Model with 2 attributes is better")
    else:
        print("Result: Model with 1 attribute is better")

    print("\nIII. Testing bayes classifier for random sample...")
    # random sample (not from dataset):
    testing_sample=np.array([12.87,4.31,1.39,31,52,1.86,5.03,.21,2.91,2.8,.75,7.64,880])
    print("Sample: ", testing_sample)
    print("Sample classified to class: ", int(bayes_classifier(samples, testing_sample)))

