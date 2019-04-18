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
        return self.data[:, 0]

    def get_labeled(self):
        return self.data