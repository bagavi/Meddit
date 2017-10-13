import numpy as np
import os
import helper, data_loader

np.random.seed(0)
filepath = '../datasets/rnaseq/20k_200_distances.npy'
print "20k"
print  os.path.isfile(filepath)
if not os.path.isfile(filepath):
    data = data_loader.load_rnaseq20k()
    random_200_arms = np.random.randint(data.shape[0], size=200)
    distance_matrix = helper.l1_dist(data, data[random_200_arms])
    np.save(filepath, distance_matrix )

print "100k"
filepath = '../datasets/rnaseq/100k_200_distances.npy'
print  os.path.isfile(filepath)
if not os.path.isfile(filepath):
    data = data_loader.load_rnaseq100k()
    random_200_arms = np.random.randint(data.shape[0], size=200)
    distance_matrix = helper.l1_dist(data, data[random_200_arms])
    np.save(filepath, distance_matrix )
