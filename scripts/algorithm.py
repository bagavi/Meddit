import collections
import matplotlib, datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import tables
import pickle, time
from multiprocessing import Pool
import cProfile, os, argparse
from scipy.sparse import hstack
import imp
import os,binascii, datetime
import multiprocessing as mp
import itertools
from sys import getsizeof
import logging, data_loader, helper
import sys
logging.basicConfig(level=logging.DEBUG,format='(%(threadName)-10s) %(message)s',)


"""
    Performs the initialization step.
    This is called only once.
    n_jobs > 1 will give errors
"""
def initialise(data, init_size, dist_func):
    n = data.shape[0]
    tmp_pos = np.array(np.random.choice(n, size=init_size, replace=False), dtype='int')
    estimate = np.mean(dist_func(data, data[tmp_pos]), axis = 1)
    return estimate

"""
    Main function
    exp_index   : A number to indicate the experiment number. Also acts as a seed
    data_loader : Function to load the data
    dataset_name: Name of the dataset. Used to save the results
    dist_func   : Function to evaluate the distances
    
"""
def UCB(arg_tuple):
    exp_index   = arg_tuple[0]
    data_loader = arg_tuple[1]
    dataset_name= arg_tuple[2]
    dist_func   = arg_tuple[3]
    sigma       = arg_tuple[4]
    verbose     = arg_tuple[5]
    
    np.random.seed(exp_index) #Random seed for reproducibility
    print "loading dataset",
    # Variable initialization
    data      = data_loader()
    n         = data.shape[0]
    Delta     = 1.0/n #Accuracy parameter. Increase this if you want to increase the speed
    num_arms  = 32 #Number of arms to be pulled in every round parallelly
    step_size = 32 #Number of distance evaluation to be performed on every arm
    lcb       = np.zeros(n, dtype='float')       #At any point, stores the mu - lower_confidence_interval
    ucb       = np.zeros(n, dtype='float')       #At any point, stores the mu + lower_confidence_interval
    T         = step_size*np.ones(n, dtype='int')#At any point, stores number of times each arm is pulled
    
    #Calculating the approximate std deviation
    sample_distance = dist_func( data[np.random.randint(n,size=2000)], data[np.random.randint(n,size=2000)] ).flatten()


    # Bookkeeping variables
    start_time = time.time()
    summary    = np.zeros(n*10)

    if exp_index==0:
        print "Calculating full summary for exp 0"
        full_summary = []
        left_over_array = [] 
        
    old_tmean  = 0
    """
        Chooses the "num_arms" arms with lowest lcb and removes the ones which have been pulled n times.
        Returns None at stopping time
    """
    def choose_arm():
        low_lcb_arms = np.argsort(lcb)[:num_arms]
        arms_to_pull = []
        arms_finished_pulling = []
        
        #Removing arms which have been pulled n times
        for arm in low_lcb_arms:
            if T[arm] >= n:
                arms_finished_pulling += [arm]
                if ucb[arm] != lcb[arm]:
#                    print exp_index, "Pull like crazy", arm
                    estimate[arm] = np.mean(dist_func(data[arm:arm+1],data))
                    T[arm]  += n
                    ucb[arm] = estimate[arm]
                    lcb[arm] = estimate[arm]
                if ucb[arm] < lcb[np.argsort(lcb)[1]]: #Exit
                    return None
            else:
                arms_to_pull += [arm]

        if len(arms_to_pull) == 0:
            return None
        return arms_to_pull


    """
        Pulls the "num_arms" arms "step_size" times. Updates the estimate, ucb, lcb
    """
    def pull_arm(arms):
        tmp_pos      = np.array( np.random.choice(n, size=step_size, replace=False), dtype='int')    
        X_arm        = data[arms]
        X_other_arms = data[tmp_pos]

        Tmean = np.mean(dist_func(X_arm,X_other_arms), axis=1)
        estimate[arms]   = (estimate[arms]*T[arms] + Tmean*step_size)/( T[arms] + step_size + 0.0 )
        T[arms]          = T[arms]+step_size
        lcb[arms]        = estimate[arms] - np.sqrt(sigma**2*np.log(1/Delta)/(T[arms]+0.0))
        ucb[arms]        = estimate[arms] + np.sqrt(sigma**2*np.log(1/Delta)/(T[arms]+0.0))

    
    #Step 1: Initialize
    estimate = initialise(data, step_size, dist_func)
    lcb      = estimate - np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    ucb      = estimate + np.sqrt(sigma**2*np.log(1/Delta)/step_size)

    
    print "running experiment ", exp_index, "with sigma", sigma
    #Step 2: Iterate
    for ind in range(n*10):
        
        #Storing the whole experiment for the first experiment
        if exp_index==0:
            order = estimate.argsort()
            full_summary += [ [order, estimate, lcb, ucb, T] ]
            left_over_array += [np.where(lcb < np.min(ucb))]
        
        
        #Choose the arms
        arms_to_pull = choose_arm()
        
        #Stop if we have found the best arm
        if arms_to_pull == None:
            logging.info("Done. Best arm = "+str(np.argmin(lcb)))
            print "Summary: Avg pulls=", T.mean(), time.time()-start_time
            break

        #Pull the arms
        pull_arm(arms_to_pull)

        #Stats
        summary[ind] = estimate.argmin()
        if T.mean() > old_tmean:
            old_tmean = T.mean() + 10
            thrown_away = (100.0*np.where(lcb > np.min(ucb))[0].shape[0])/n
            if verbose:
                logging.info(str(exp_index)+" Thrown away "+" "+str(thrown_away)+" "+str(T.mean())+" "+str(T.std()) ) 
                
    if exp_index==0:
        print "Saving full summary for exp 0"
        filename = '../experiments/figure_data/'+dataset_name+'_sample.pkl'
        with open(filename,'wb') as f:
            pickle.dump([full_summary, left_over_array, T] ,f)
            
    filename = '../experiments/'+dataset_name+'/meddit/'+str(exp_index)+'.pkl'
    with open(filename,'wb') as f:
        pickle.dump([summary, T.mean()] ,f)


ap = argparse.ArgumentParser(description="Reproduce the experiments in the manuscript")
ap.add_argument("--dataset",  help="Name of the dataset eg. rnaseq20k, netflix100k")
ap.add_argument("--num_exp",  help="max size of any split file.", type=int, default=32 )
ap.add_argument("--num_jobs", help="Num of parallel experiments", type=int, default=32 )
ap.add_argument("--verbose",  help="Running outputs", type=bool, default=False )

args = ap.parse_args()

num_jobs   = args.num_jobs
num_trials = args.num_exp
dataset    = args.dataset        
verbose    = args.verbose        

if dataset   == 'rnaseq20k':
    data_loader = data_loader.load_rnaseq20k
    dist_func   = helper.l1_dist
    sigma       = 0.25
elif dataset == 'rnaseq100k':
    data_loader = data_loader.load_rnaseq100k
    dist_func   = helper.l1_dist
    sigma       = 0.25
elif dataset == 'netflix20k':
    data_loader = data_loader.load_netflix20k
    dist_func   = helper.cosine_dist
    sigma       = 0.2
elif dataset == 'netflix100k':
    data_loader = data_loader.load_netflix100k
    dist_func   = helper.cosine_dist
    sigma       = 0.2
elif dataset == 'mnist':
    data_loader = data_loader.load_mnist
    dist_func   = helper.l2_dist
                   
print "Running", num_trials, "experiments on ", num_jobs, "parallel jobs", "on dataset", dataset
arg_tuple =  itertools.product(range(num_trials), [data_loader], [dataset], [dist_func], [sigma], [verbose] )
pool      = mp.Pool(processes=num_jobs)
pool.map(UCB, arg_tuple)
