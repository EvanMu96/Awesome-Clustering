"""
author: Komoriii
EE6435 homework5
"""
import numpy as np
import argparse
from dataloader import Dataloader
from random import uniform

parser = argparse.ArgumentParser(description="Kmeans for 1 dimensional data")
parser.add_argument("--vis", dest="vis", type=bool, default=False)
args = parser.parse_args()

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
            for cent in range(2):
                pts = data[np.nonzero(cluster_table[:, 0].A==cent)[0]]
                centroids[cent] = np.mean(pts)
        #return centroid, cluster_table
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
        return corr/n
            
# unit test
if __name__ == "__main__":
    # load dataset generated
    dloader = Dataloader("data_close.npy")
    # get unlabeled dataset
    data = dloader.get_unlabeled()
    # initialization. The initial point is between max and min of data
    max_data, min_data = np.max(data), np.min(data)
    initial_points = [uniform(min_data, max_data), uniform(min_data, max_data)]
    print("The initial point is {}".format(initial_points))
    km1d = KMeans1D(initial_points)
    data = dloader.get_labeled()
    print(km1d.eval(data))

    
    
