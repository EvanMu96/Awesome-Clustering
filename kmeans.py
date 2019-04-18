"""
author: Komoriii
EE6435 homework5
"""
import numpy as np
import argparse
from dataloader import Dataloader

parser = argparse.ArgumentParser(description="Kmeans for 1 dimensional data")
parser.add_argument("--initial", dest="inip", type=str, default="2,-2")
parser.add_argument("--vis", dest="vis", type=bool, default=True)
args = parser.parse_args()

initial_points = [float(i) for i in args.inip.split(",")]


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
                    if distJI < minDist:
                        minDist = distJI; minIndex = j
                    cluster_table[i, :] = minIndex, minDist
                print(centroids)
                # recalculate centroids
            for cent in range(2):
                pts = data[np.nonzero(cluster_table[:, 0].A==cent)[0]]
                centroids[cent] = np.mean(pts)
        #return centroid, cluster_table
        return centroids

if __name__ == "__main__":
    print(initial_points)
    # load dataset generated
    dloader = Dataloader("data.npy")
    # get unlabeled dataset
    data = dloader.get_unlabeled()
    print(data.shape)
    km1d = KMeans1D(initial_points)
    print(km1d.cluster(data))
    
    
