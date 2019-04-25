"""
author: Komoriii
EE6435 homework5
preprocess the labeled data to unlabeled or other format.
"""
import numpy as np

class Dataloader:
    def __init__(self, data_path):
        self.data = np.load(data_path)
    
    def get_unlabeled(self):
        return self.data[:, 0].reshape((self.data.shape[0], 1))

    def get_labeled(self):
        return self.data

if __name__ == "__main__":
    dloader = Dataloader("data.npy")
    data = dloader.get_labeled()
    print(data.shape, ' labeled')
    data = dloader.get_unlabeled()
    print(data.shape, ' unlabeled')