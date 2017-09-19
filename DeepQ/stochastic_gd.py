import numpy as np
from scipy.special import expit
# test a  3*3 neural network

def derivative(x):
    return expit(x) * (1 - expit(x))

def converge(RMSE, number, threshold):
    if RMSE < threshold:
        number += 1
    else:
        number = 0
    if number > 10:
        return True, number
    else: 
        return False, number
        
data = np.array(
[[0,0,0,.1],
[1,1,1,0],
[1,0,0,.5],
[0,1,1,.4],
[0,1,0,.3],
[1,0,1,.6]])
training = data[:,0:3]
training_res = data[:, 3]

N_Bias = 1
N_hidden = 3
gamma = 0.4
alpha = 1

N_datapoints = training.shape[0]
N_dimension = training.shape[1]
N_dimension2 = 1
training_res = training_res.reshape((N_datapoints,N_dimension2))

N_sample = np.int(np.round(0.6 * N_datapoints))

training_in_all = np.hstack((training, np.ones((N_datapoints,1)) ))

while True:
    W1 = np.random.rand(N_dimension+N_Bias,N_hidden) # Weights from input to hidden layer
    W2 = np.random.rand(N_hidden+N_Bias,N_dimension2) # weights from hidden to output 
    iteration = 0
    number = 0
    while True:
        # Update W1 and W2 each step until converged
        sample_index = np.random.choice(range(N_datapoints), N_sample, replace=False)
        training_in = np.hstack((training[sample_index], np.ones((N_sample,1)) ))
        
        hidden_in = np.dot(training_in, W1) 
        hidden_out = expit(hidden_in)
        hidden_out_b = np.hstack((hidden_out, np.ones((N_sample,1))))
        hidden_dev = derivative(hidden_in)
        out_in = np.dot(hidden_out_b, W2) 
        out_out = expit(out_in)
        out_dev = derivative(out_in)
        out_error = out_out - training_res[sample_index]
        
        iteration += 1
        RMSE = (out_error**2).sum() * 0.5
        conv, number = converge(RMSE, number, 0.005)
        if conv or iteration>5000: 
            print RMSE,iteration
            break
    
        out_delta = out_error * out_dev
        W2_g = np.dot(hidden_out_b.T, out_delta)
        W1_g = np.dot(training_in.T, np.dot(out_delta, W2.T)[:, :N_dimension] * hidden_dev)   
        
        W1 -= W1_g * gamma
        W2 -= W2_g * gamma
        gamma *= alpha
    if conv:
        break
"""
training_in = np.hstack((training, np.ones((N_datapoints,1)) ))
hidden_in = np.dot(training_in, W1) 
hidden_out = expit(hidden_in)
hidden_out_b = np.hstack((hidden_out, np.ones((N_datapoints,1))))
hidden_dev = derivative(hidden_in)
out_in = np.dot(hidden_out_b, W2) 
out_out = expit(out_in)
out_dev = derivative(out_in)
out_error = out_out - training_res
RMSE = (out_error**2).sum() * 0.5    
print RMSE
"""
    