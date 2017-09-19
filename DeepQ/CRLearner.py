"""
Template for implementing CRLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
from scipy.special import expit

class CRLearner(object):

    def __init__(self, \
        num_dimensions = 3, \
        num_actions = 4, \
        alpha = 1, \
        gamma = 0.95,\
        rar = 0.99, \
        radr = 0.9999, \
        verbose = False):

        self.num_dimensions = num_dimensions
        self.num_actions = num_actions
	self.verbose = verbose
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.s = 0
        self.a = 0
        self.D = []
        self.D_num = 0
        self.actions = {0:[0,0],1:[0,1],2:[1,0],3:[1,1]}
        self.reached = False

        self.Q = DeepQ(num_actions = self.num_actions, \
        N_hidden = 8, \
        N_dimension1 = num_dimensions**2+2, \
        N_dimension2 = 1)      
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.D.append([self.s, self.a, r, s_prime])
        self.D_num += 1
        if r == 1: self.reached = True
        
        for i in xrange(20):
            s_t, a_t, r_t, s_t1 = self.D[np.random.randint(self.D_num)]
            phi = np.hstack((self.discretize(s_t), np.array(self.actions[a_t]) ))
            if r_t == 1: 
                Q = 1
                trained = np.hstack((phi, Q))
                for j in xrange(10): self.Q.train(trained) 
            else: 
                Q_best_learn, action_best_learn = self.Q.best(self.discretize(s_t1), False) 
                Q_best_learn = Q_best_learn * 21 - 20
                Q = r_t + self.gamma * Q_best_learn
                Q = max(0, (Q+20)/21.0)
                self.Q.train(np.hstack((phi, Q))) 
                
        
        Q_best_current, action_best_current = self.Q.best(self.discretize(s_prime), self.verbose)
        action_random = rand.randint(0, self.num_actions-1)
        if self.reached:
            action = np.random.choice([action_random, action_best_current], p = [self.rar, 1-self.rar])
            self.rar *= self.radr
        else:
            action = action_random
        
        self.s = s_prime
        self.a = action        
                
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action
    def discretize(self, state):
        """
        state is a 2 items list
        return a N*N list; N is num_dimensions
        """
        dimension = self.num_dimensions-1
        row = min(max(0, np.int(np.round(state[0] * dimension))), dimension)
        col = min(max(0, np.int(np.round(state[1] * dimension))), dimension)

        state_discrete = np.zeros(self.num_dimensions**2)
        state_discrete[row * self.num_dimensions + col] = 1
        return state_discrete

class DeepQ(object):
    
    def __init__(self, \
        N_Bias = 1, \
        N_hidden =12, \
        N_dimension1 = 3, \
        N_dimension2 = 1, \
        num_actions = 4, \
        sample_rate = 0.6, \
        gamma = 0.5, \
        threshold = 0.01):
        
        self.N_Bias = N_Bias
        self.N_hidden = N_hidden
        self.gamma = gamma
        self.num_actions = num_actions
        self.N_dimension1 = N_dimension1 # input demension
        self.N_dimension2 = N_dimension2 # output demension
        self.sample_rate = sample_rate
        self.data = np.zeros((0,N_dimension1+N_dimension2))
        self.threshold = threshold

        self.W1 = np.random.rand(self.N_dimension1+self.N_Bias,self.N_hidden) # Weights from input to hidden layer
        self.W2 = np.random.rand(self.N_hidden+self.N_Bias,self.N_dimension2) # weights from hidden to output 
        self.W1_new = self.W1.copy()
        self.W2_new = self.W2.copy()
        self.train_time = 0
                
    def train(self,datapoint):
        """
        add an new entry, the datapoint should be a (N_dimension+1) array
        """
        self.train_time += 1
        if self.train_time %50 == 0:
            self.W1 = self.W1_new.copy()
            self.W2 = self.W2_new.copy()
        N_sample = 1
        datapoint = datapoint.reshape((1, self.N_dimension1+self.N_dimension2))
        training = datapoint[:, 0:self.N_dimension1]
        training_res = datapoint[:, self.N_dimension1:self.N_dimension1+self.N_dimension2]
        training_in = np.hstack((training, np.ones((N_sample,1)) ))
        hidden_in = np.dot(training_in, self.W1) 
        hidden_out = expit(hidden_in)
        hidden_out_b = np.hstack((hidden_out, np.ones((N_sample,1))))
        hidden_dev = self.derivative(hidden_in)
        out_in = np.dot(hidden_out_b, self.W2) 
        out_out = expit(out_in)
        out_dev = self.derivative(out_in)
        out_error = out_out - training_res  
        RMSE = (out_error**2).sum() * 0.5
    #    print RMSE   

        out_delta = out_error * out_dev
        W2_g = np.dot(hidden_out_b.T, out_delta)
        W1_g = np.dot(training_in.T, np.dot(out_delta, self.W2.T)[:, :self.N_hidden] * hidden_dev)   
        
                                
        self.W1_new = self.W1_new - W1_g * self.gamma
        self.W2_new = self.W2_new - W2_g * self.gamma

    
    def query(self, test):
        """
        test is a N_dimension1 size array
        """
        training = test.reshape((1, self.N_dimension1))
        N_sample = 1
        training_in = np.hstack((training, np.ones((N_sample,1)) ))
        hidden_in = np.dot(training_in, self.W1) 
        hidden_out = expit(hidden_in)
        hidden_out_b = np.hstack((hidden_out, np.ones((N_sample,1))))
        out_in = np.dot(hidden_out_b, self.W2) 
        out_out = expit(out_in)        
        return out_out
            
    def best(self, state, verbose):
        """
        return the best action and the best action's Q value under current state
        """
        actions = {0:[0,0],1:[0,1],2:[1,0],3:[1,1]}
        action_list = np.zeros(4)
        for action in actions:
            test_in = np.array( list(state) + actions[action] )
            action_list[action] = self.query(test_in)   
        if verbose:
            print action_list
        action_best = action_list.argmax()
        return (action_list[action_best], action_best)
        
    def derivative(self,x):
        return expit(x) * (1 - expit(x))
    
    def converge(self, RMSE, number):
        if RMSE < self.threshold:
            number += 1
        else:
            number = 0
        if number > 10:
            return True, number
        else: 
            return False, number
            
if __name__=="__main__":
    print "Well done!"
