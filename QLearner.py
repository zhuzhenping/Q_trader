"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        self.num_states = num_states
        self.verbose = verbose
        self.num_actions = num_actions
        self.rar = rar
        self.radr = radr
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.R = np.zeros((self.num_states,self.num_actions))
        self.Q = np.zeros((self.num_states,self.num_actions))
        self.mem = []
        self.mem_num = 0
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        s = int(s)
        self.s = s
        action_random = rand.randint(0, self.num_actions-1)
        action_best = self.Q[s, :].argmax()
        action = np.random.choice([action_random, action_best], p = [self.rar, 1-self.rar])
        self.rar *= self.radr
        self.a = action
        if self.verbose: print "s =", s,"a =",self.a
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        s_prime = int(s_prime)
        if self.dyna > 0:
            self.mem.append([self.s, self.a, s_prime, r])
            self.mem_num += 1

        self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + self.alpha * r 
        self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + \
        self.alpha * (r + self.gamma * self.Q[s_prime, :].max())
        if self.dyna > 0:
            if self.mem_num >= 20:
                for i in xrange(self.dyna):
                    index = rand.randint(0, self.mem_num-1)
                    s_d, a_d, sp_d, r_d = self.mem[index]
                    self.Q[s_d, a_d] = (1-self.alpha) * self.Q[s_d, a_d] + \
                    self.alpha * (r_d + self.gamma * self.Q[sp_d, :].max())
        action_best = self.Q[s_prime, :].argmax()
        action_random = rand.randint(0, self.num_actions-1)
        action = np.random.choice([action_random, action_best], p = [self.rar, 1-self.rar])
        self.rar *= self.radr
        self.s = s_prime
        self.a = action
        if self.verbose: print "s =", s_prime,"a =",self.a,"r =",r   
        return action
    def author(self):
        return 'yjiao43'
if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
