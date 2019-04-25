"""
author: Komoriii
EE6435 homework5
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats
import time
import seaborn as sns
from math import sqrt, log, exp, pi
from random import uniform
from dataloader import Dataloader


class Gaussian:
    "k=2, gaussian mixture model"

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y

    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

class GaussianMixture:
    "Model mixture of two univariate Gaussians and their EM estimation"
    def __init__(self, data, sigma_min=.1, sigma_max=1, mix=.5):
        mu_min, mu_max=min(data), max(data)
        self.data = data
        self.one = Gaussian(uniform(mu_min, mu_max), 
                            uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max), 
                            uniform(sigma_min, sigma_max))
        self.mix = mix

    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        self.labels = []
        # compute weights
        self.loglike = 0. # = log(p = 1)
        #self.p = 1
        for datum in self.data:
            # unnormalized weights
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.two.pdf(datum) * (1. - self.mix)
            # compute denominator
            den = wp1 + wp2
            # normalize
            wp1 /= den
            wp2 /= den
            # posterior probability
            post_prob = wp1 + wp2
            self.labels.append(np.argmax([wp1, wp2]))
            # add into loglike
            self.loglike += log(post_prob)
            # yield weight tuple
            yield (wp1, wp2)

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        # compute denominators
        (left, rigt) = zip(*weights)
        one_den = sum(left)
        two_den = sum(rigt)
        # compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(rigt, data))
        # compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(rigt, data)) / two_den)
        # compute new mix
        self.mix = one_den / len(data)

    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        for i in range(1, N+1):
            self.Mstep(self.Estep())
            if verbose:
                print('{0:2} {1}'.format(i, self))
        self.Estep() # to freshen up self.loglike

    def pdf(self, x):
        return (self.mix)*self.one.pdf(x) + (1-self.mix)*self.two.pdf(x)

    def eval(self, data):
        p_labels = self.labels
        gt_labels = data[:, 1]
        corr = 0
        n = len(p_labels)
        for i in range(n):
            if p_labels[i] == gt_labels[i]:
                corr += 1
        if corr/n < 0.5:
            return 1 - corr/n
        else:
            return corr/n
     
    def __repr__(self):
        return 'GaussianMixture({0}, {1}, mix={2.03})'.format(self.one, 
                                                              self.two, 
                                                              self.mix)

    def __str__(self):
        return 'Mixture: {0}, {1}, mix={2:.03})'.format(self.one, 
                                                        self.two, 
                                                        self.mix)
        



if __name__ == "__main__":
    # Find best Mixture Gaussian model
    dloader = Dataloader(sys.argv[1])
    data = dloader.get_unlabeled()
    n_iterations = 10
    best_mix = None
    best_loglike = float('-inf')
    start = time.clock()
    print('Computing best model with random restarts...\n')
    mix = GaussianMixture(data)
    for _ in range(n_iterations):
        try:
            mix.iterate()
            if mix.loglike > best_loglike:
                best_loglike = mix.loglike
                best_mix = mix
        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
            pass
    print('if there are some NaN value please try again')
    print("Two means of models are {} and {}".format(best_mix.one.mu, best_mix.two.mu))
    print("Two stdvars of models are {} and {}".format(best_mix.one.sigma, best_mix.two.sigma))
    data = dloader.get_labeled()
    print("The accuracy is: {}".format(best_mix.eval(data)))
    print("Total running time: {}".format(time.clock()-start))
    