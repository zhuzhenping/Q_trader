import numpy as np
from scipy.special import expit

class DeepQ(object):
    
    def __init__(self, \
        N_Bias = 1, \
        N_hidden = 3, \
        N_dimension1 = 3, \
        N_dimension2 = 1, \
        num_actions = 4, \
        sample_rate = 0.6, \
        gamma = 0.6, \
        threshold = 0.005):
        
        self.N_Bias = N_Bias
        self.N_hidden = N_hidden
        self.gamma = gamma
        self.num_actions = num_actions
        self.N_dimension1 = N_dimension1 # input demension
        self.N_dimension2 = N_dimension2 # output demension
        self.N_datapoints = 0
        self.sample_rate = sample_rate
        self.data = np.zeros((0,N_dimension1+N_dimension2))
        self.threshold = threshold

        self.W1 = np.random.rand(self.N_dimension1+self.N_Bias,self.N_hidden) # Weights from input to hidden layer
        self.W2 = np.random.rand(self.N_hidden+self.N_Bias,self.N_dimension2) # weights from hidden to output 
        
    def train(self,datapoint):
        """
        add an new entry, the datapoint should be a (N_dimension+1) array
        """
        self.N_datapoints += 1
        N_sample = np.int(np.round(0.6 * self.N_datapoints))
        self.data = np.concatenate((self.data, datapoint))
        while True:
            iteration = 0
            number = 0
            while True:
                iteration+=1
                sample_index = np.random.choice(range(self.N_datapoints), N_sample, replace=False)
                training = self.data[sample_index, 0:self.N_dimension1]
                training_res = self.data[sample_index, self.N_dimension1:self.N_dimension1+self.N_dimension2]
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
                conv, number = self.converge(RMSE, number)
                if conv or iteration>3000: 
                    print RMSE,iteration
                    break
                out_delta = out_error * out_dev
                W2_g = np.dot(hidden_out_b.T, out_delta)
                W1_g = np.dot(training_in.T, np.dot(out_delta, self.W2.T)[:, :self.N_hidden] * hidden_dev)   
                
                self.W1 -= W1_g * self.gamma
                self.W2 -= W2_g * self.gamma
            if conv: break
            else:
                self.W1 = np.random.rand(self.N_dimension1+self.N_Bias,self.N_hidden) # Weights from input to hidden layer
                self.W2 = np.random.rand(self.N_hidden+self.N_Bias,self.N_dimension2) # weights from hidden to output 
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
            
    def best(self, state):
        """
        return the best action and the best action's Q value under current state
        """
        action_list = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            test_in = np.hstack((state, np.array([action*1.0/(self.num_actions-1)]) ))
            action_list[action] = self.query(test_in)
            
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
            
            
testQ=DeepQ(N_Bias = 1, \
        N_hidden = 3, \
        N_dimension1 = 3, \
        N_dimension2 = 1, \
        num_actions = 4, \
        sample_rate = 0.6, \
        gamma = 0.6, \
        threshold = 0.005)