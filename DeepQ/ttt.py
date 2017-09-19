import numpy as np
sig = 0.05
point = (0.74, 0.26)
k_size = 11
row = 1/sig/np.sqrt(2*np.pi)*np.exp(-(np.arange(k_size)/np.float(k_size-1)-point[0])**2/(2*sig**2))
col = 1/sig/np.sqrt(2*np.pi)*np.exp(-(np.arange(k_size)/np.float(k_size-1)-point[1])**2/(2*sig**2))
res = np.dot(row.reshape((k_size,1)), col.reshape((1,k_size)))
res = res/res.sum()
res[res< 1e-4] = 0