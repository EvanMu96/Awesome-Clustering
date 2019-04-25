"""
author: Komoriii
EE6435 homework5
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from math import sqrt, log, exp, pi

class Gaussian:
    "Model univariate Gaussian"
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    # probability density function
    def pdf(self, datum):
        u = (datum - self.mu)/abs(self.sigma)
        y = (1/sqrt(2*3.1415)*abs(self.sigma))*exp(-u*u/2)
        return y
    
    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

class GaussianMixture:
    def __init__(self, data, mu_min=min(data), mu_max=max(data), sigma_min=.1, sigma_max=1, mix=.5):
        self.data =data
        self.one = Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max))
        self.mix = mix
        
    def Estep(self):
        self.loglike= 0.
        # compute weights
        for datum in self.data:
            # unnormalized weights
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.one.pdf(datum) * (1. - self.mix)
            den = wp1 + wp2
            wp1 /= den
            wp2 /= den
            self.loglike += log(wp1 + wp2)
            # yield weight tuple
            yield (wp1, wp2)
            
    def Mstep(self, weights):
        # updata model parameters
        (left, right) = zip(*weights)
        one_den = sum(left)
        two_den = sum(right)
        # compute the new means
        self.one.mu = sum(w * d/ one_den for (w, d) in zip(left, data))
        self.two.mu = sum(w * d/ two_den for (w, d) in zip(left, data))
        # compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(right, data)) / two_den)
        
        # compute new mix
        self.mix = one_den / len(data)
        
    def iterate(self, N=1, verbose=False):
        # initialize
        n = 0
        "Perform N iterations, then compute log-likelihood"
        while n < N:
            n += 1
            self.Mstep(self.Estep())
            if verbose:
                print('{0:2} {1}'.format(i, self))
            self.Estep() # to freshen uo self.loglike
    
    def pdf(self, x):
        return (self.mix) * self.one.pdf(x) + (1-self.mix)*self.twp.pdf(x)
    
    def __repr__(self):
        return "GaussianMixture({0}, {1}, mix={2.03})".format(self.one, self.two, self.mix)
    
    def __str__(self):
        return "Mixture: {0}, {1}, mix={2:.03}".format(self.one, self.two, self.mix)
        
if __name__ == "__main__":
    n_iterations = 5
    best_mix = None
    best_loglike = float('-inf')
    mix = GaussianMixture(data)
    for i in range(n_iterations):
        try:
            mix.iterate(verbose=True)
            if mix.loglike > best_loglike:
                # iterate to find the best
                print('!')
                best_loglike = mix.loglike
                best_mix = mix
                print(mix.loglike)
                print(best_mix)
        # sometimes it might be some zeroDivision Error, just skip that
        except (ZeroDivisionError, ValueError, RuntimeWarning):
            pass
    print("Training done")