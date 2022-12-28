# Testing id3 algorithm
# author: Radosław Pietkun


from id3 import id3
from models import classify_sample, test_model, find_best_model
import numpy as np


# Load data, skip 2.column (passengers names) and 1. row (attributes names)
U = np.loadtxt("titanic.csv", delimiter=",", dtype="str", skiprows=1, usecols=(0,1,3,4,5,6,7))


# Parameters
class_column_id = 0  # id of column which contains classes
training_part = 0.8  # training part of dataset (the rest is testing part)
n_cross = 10  # n-cross validation param
max_depth = 3  # max depth of tree

num_samples = U.shape[0]
num_cols = U.shape[1]
num_attr = U.shape[1] - 1

# Prepare data for analysis (some columns need to be adjusted):

# 'Age' - 3. column
age_groups = {
    "0-15": [0, 15],  # 0 included, 15 excluded
    "15-25": [15, 25],
    "25-35": [25, 35],
    "35-45": [35, 45],
    "45+": [45, 100]
}
for i in range(num_samples):
    for j in list(age_groups.keys()):
        if (float((U[i,3]))) >= age_groups[j][0] and (float((U[i,3]))) < age_groups[j][1]:
            U[i,3] = j
            break

# 'Siblings/Spouses Aboard' - 4. column
# 3 options are differentiated: 0, 1, 2 or more
for i in range(num_samples):
    if int(U[i,4]) >= 2:
        U[i,4] = '2+'

# 'Parents/Children Aboard' - 5.column
# 3 options are differentiated: 0, 1, 2 or more
for i in range(num_samples):
    if int(U[i,5]) >= 2:
        U[i,5] = '2+'

# 'Fare' - 6. column
# 5 options are differentiated: [1,10), [10,20), [20,30), [30,60), 60 or more
fare_groups = {
    "0-10": [0, 10],  # 0 included, 10 excluded
    "10-20": [10, 20],
    "20-30": [20, 30],
    "30-60": [30, 60],
    "60+": [60, 1000]
}
for i in range(num_samples):
    for j in list(fare_groups.keys()):
        if (float((U[i,6]))) >= fare_groups[j][0] and (float((U[i,6]))) < fare_groups[j][1]:
            U[i,6] = j
            break

# Now 'U' dataset is ready to be used in classification

# create training and testing dataset (validating dataset is included in training set and will be extracted in find_best_model() function)
U_training = U[0:int(training_part*num_samples),:]
U_testing = U[int(training_part*num_samples):,:]

D = {}  # dictionary for attributes: key - attr id (column id in 'U' table, value - list of all possible values for this attribute
print("Attributes (all samples included):")
for i in range(num_cols):
    if i != class_column_id:  # skip class column
        attr_val, counts = np.unique(U[:, i], return_counts=True)
        D[i] = list(attr_val)
        print("attr_id={}, attr_val={}, counts={}".format(i, attr_val, counts))


print("\nI. Finding best model...")
root = find_best_model(U_training, n_cross, D, class_column_id, max_depth)
print("Testing best model...")
missclass_count, classify_failures, num_test_samples = test_model(root, U_testing, class_column_id)
print("Best model testing results: missclass_count: {}, classify_failures: {}, "
      "num_testing_samples: {}".format(missclass_count, classify_failures, num_test_samples))
print("Best model accuracy on testing set: {:.2f}% ".format( (1-(missclass_count+classify_failures)/num_test_samples)*100 ) )


print("\nII. Try to classify sample")
# Accepted values for attributes:
# 0 - Survived: ["0","1"]
# 1 - Pclass: ["1","2","3"]
# 2- Sex: ["male", "female"]
# 3- Age: ["0-15", "15-25", "25-35", "35-45", "45+"]
# 4- Siblings/Spouses Aboard: ["0","1","2+"]
# 5- Parents/Children Aboard: ["0","1","2+"]
# 6- Fare: ["0-10", "10-20", "20-30", "30-60", "60+"]
testing_sample = np.array(['?', '3', 'male', '15-25', '1', '0', '10-20'])
#testing_sample = np.array(['?', '1', 'female', '35-45', '0', '0', '20-30'])
print("testing_sample:", testing_sample)
classified_class = classify_sample(root, testing_sample)
print("classified_class: {}".format(classified_class))
if classified_class == str(0):
    print("I'd not have survived :(")
elif classified_class == str(1):
    print("I'd have survived :)")

