import numpy as np
import csv
import pickle

# Separate classes
ones = []
zeros = []
with open('data/creditcard.csv', 'r') as f:
    data = csv.reader(f, delimiter=',')
    for line in data:
        if line[-1] == '0':
            zeros.append(line[:-1] + [1, 0])
        elif line[-1] == '1':
            ones.append(line[:-1] + [0, 1])

training_set = np.array(ones[:-93] + zeros[:-50000])
training_set_ones = np.array(ones[:-93])
training_set_zeros = np.array(zeros[:-50000])

test_set = np.array(ones[-93:] + zeros[-50000:])
test_set_ones = np.array(ones[-93:])
test_set_zeros = np.array(zeros[-50000:])

#np.random.shuffle(test_set)
#np.random.shuffle(training_set)

test_set_x = test_set[:, :-2]
test_set_y = test_set[:, -2:]

training_set_x = training_set[:, :-2]
training_set_y = training_set[:, -2:]

data_set = [training_set_x, training_set_y,
            test_set_x, test_set_y]
data_set_10 = [training_set_ones, training_set_zeros,
               test_set_ones, test_set_zeros]

with open('data/data_set.pickle', 'wb') as handle:
    pickle.dump(data_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/data_set_10.pickle', 'wb') as handle:
    pickle.dump(data_set_10, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
with open('data/data_set.pickle', 'rb') as handle:
    b = pickle.load(handle)
"""
