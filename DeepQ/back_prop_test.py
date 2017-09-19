import numpy as np
from scipy.special import expit
# test a  3*3 neural network

def derivative(x):
    return expit(x) * (1 - expit(x))

def forward(dataset, W1, W2):
    training_in = np.hstack((dataset, np.ones((dataset.shape[0],1)) ))
    hidden_in = np.dot(training_in, W1) 
    hidden_out = expit(hidden_in)
    hidden_out_b = np.hstack((hidden_out, np.ones((dataset.shape[0],1))))
    out_in = np.dot(hidden_out_b, W2) 
    out_out = expit(out_in)
    return out_out
            
data = np.array(
[[0.5,0.3,0, 0.1],
[1,1,1, 0],
[1,0,0, 0.5],
[0,1,1, 0.4],
[0.1,0.9,0.2, 0.88],
[0.8,0,0.9, 0.9]
])
N_dimension1 = 3
N_dimension2 = 1

training = data[:,0:N_dimension1]
training_res = data[:, N_dimension1:]

N_Bias = 1
N_hidden = 10
gamma = 0.8
alpha = 1

N_datapoints = training.shape[0]


training_in = np.hstack((training, np.ones((N_datapoints,1)) ))
threshold = 0.0001

while True:
    iteration = 0
    W1 = np.random.rand(N_dimension1+N_Bias, N_hidden) # Weights from input to hidden layer
    W2 = np.random.rand(N_hidden+N_Bias, N_dimension2) # weights from hidden to output 
    while True:
        hidden_in = np.dot(training_in, W1) 
        hidden_out = expit(hidden_in)
        hidden_out_b = np.hstack((hidden_out, np.ones((N_datapoints,1))))
        hidden_dev = derivative(hidden_in)
        out_in = np.dot(hidden_out_b, W2) 
        out_out = expit(out_in)
        out_dev = derivative(out_in)
        out_error = out_out - training_res
        
        iteration += 1
        RMSE = (out_error**2).sum() * 0.5
        if RMSE < threshold or iteration > 10000: 
            print (RMSE,iteration)
            break
        
        out_delta = out_error * out_dev
        W2_g = np.dot(hidden_out_b.T, out_delta)
        W1_g = np.dot(training_in.T, np.dot(out_delta, W2.T)[:, :N_hidden] * hidden_dev)   
        
        W1 -= W1_g * gamma
        W2 -= W2_g * gamma
        gamma *= alpha
    if RMSE < threshold:
        break



print forward(training, W1, W2)
    