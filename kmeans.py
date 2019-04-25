"""
author: Komoriii
EE6435 homework5
"""
import numpy as np
import time
import sys
from dataloader import Dataloader
from random import uniform


# for 1d data point, l1 distance/ l2 distance is equal :)
def l2distance(a, b):
    return np.abs(a-b)


class KMeans1D:
    def __init__(self, initial_points, iteration_number=30):
        self.ip = initial_points
        self.k = len(initial_points)
        self.it_number = iteration_number


    def cluster(self, data):
        m = data.shape[0]
        cluster_table = np.mat(np.zeros((m, 2)))
        centroids = self.ip
        std_var = [0, 0]
        for iter_num in range(self.it_number):
            for i in range(m):
                minDist = np.inf; minIndex = -1
                for j in range(self.k):
                    distJI = l2distance(data[i], centroids[j])
                    # allocate to clusters
                    if distJI < minDist:
                        minDist = distJI; minIndex = j
                    cluster_table[i, :] = minIndex, minDist
                # print(centroids)
                # recalculate centroids
            old_centroids = centroids
            for cent in range(2):
                pts = data[np.nonzero(cluster_table[:, 0].A==cent)[0]]
                centroids[cent] = np.mean(pts)
                std_var[cent] = np.std(pts)
                #print(centroids)
            # stop criterion
            if old_centroids == centroids:
                break
        #return centroid, cluster_table
        for cent in range(2):
            print("Cluster {0}: the mean is {1}, the standard var is {2}".format(cent, centroids[cent], std_var[cent]))

        return centroids, cluster_table
    
    # predict as a classification task
    def _predict_label(self, data):
        _, cls_table = self.cluster(data)
        return cls_table[:, 0]

    # calculate accuracy
    def eval(self, data):
        p_labels = self._predict_label(data[:, 0])
        gt_labels = data[:, 1]
        corr = 0
        n = p_labels.shape[0]
        for i in range(n):
            if p_labels[i] == gt_labels[i]:
                corr += 1
        if corr/n < 0.5:
            return 1 - corr/n
        else:
            return corr/n
            
# unit test
if __name__ == "__main__":
    
    # load dataset generated
    dloader = Dataloader(sys.argv[1])
    # get unlabeled dataset
    data = dloader.get_unlabeled()
    # initialization. The initial point is between max and min of data
    max_data, min_data = np.max(data), np.min(data)
    initial_points = [uniform(min_data, max_data), uniform(min_data, max_data)]
    print("The initial point are {}".format(initial_points))
    start = time.clock()
    km1d = KMeans1D(initial_points)
    data = dloader.get_labeled()
    print("The accuracy is: {}".format(km1d.eval(data)))
    print("Total running time is {}".format(time.clock() - start))
    print("If the result is far from 1.0. It means that the random number is too strange. Please re-run this program")
    """
    data = np.array([1,2,3,5,7])
    # initialization. The initial point is between max and min of data
    max_data, min_data = np.max(data), np.min(data)
    initial_points = [uniform(min_data, max_data), uniform(min_data, max_data)]
    print("The initial point is {}".format(initial_points))
    start = time.clock()
    km1d = KMeans1D(initial_points)
    km1d.cluster(data)
    print("Total running time is {}".format(time.clock() - start))
    """
    
    
