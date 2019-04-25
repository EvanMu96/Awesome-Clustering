"""
author: Komoriii
EE6435 homework5
"""

import numpy as np
import numpy.random as rdm
import argparse

parser = argparse.ArgumentParser(description="Input the data generation parameters")
parser.add_argument("--means", dest="means", type=str)
parser.add_argument("--devs", dest="devs", type=str)
parser.add_argument("--sizes", dest="sizes", type=str)
parser.add_argument("--vis", dest="vis", type=bool, default=False)
args = parser.parse_args()


# take the variables out of the namespace
means = [float(i) for i in args.means.split(',')]
devs = [float(i) for i in args.devs.split(',')]
sizes = [int(i) for i in args.sizes.split(',')]

# must have the same dimension number
assert(len(means) == len(devs))
assert(len(devs) == len(sizes))


# main function
if __name__ == "__main__":
    
    # this should be a non supervision data, however I need to calculate an accuracy in question 3...
    part1 = rdm.normal(means[0], devs[0], sizes[0])
    label1 = np.zeros_like(part1)
    part1 = np.vstack([part1, label1])
    part2 = rdm.normal(means[1], devs[1], sizes[1])
    laebl2 = np.ones_like(part2)
    part2 = np.vstack([part2, laebl2])
    # horizontally stack two array and shuffle
    data = np.hstack([part1, part2])
    data = data.T
    rdm.shuffle(data)
    if args.vis == True:
        import matplotlib.pyplot as plt
        # this is a hack to plot 1d data with 2d scatter plot
        plt.plot(data, np.zeros_like(data), 'o')
        plt.show()
    #save data to npy format. For convention use the row number as data point number
    print(data)
    np.save("ds_new.npy", data)
    print("The data have been saved to data.npy")
    
    
    
    
