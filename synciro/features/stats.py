import numpy as np

"""
    Empirical Cumulative Distribution Function
"""
def ecdf(input_array, extend=False, topbot=[0,255]):
    if len(input_array.shape) < 2: raise ValueError("Expecting numpy array with shape minimum of 2, assuming last dimension is the channel.")
    
    x = input_array.reshape(-1,input_array.shape[-1]).swapaxes(1,0).copy()
    x.sort(axis=1)
    
    if extend: 
        x[:,0] = np.minimum(x[:,0], topbot[0])
        x[:,-1] = np.maximum(x[:,0], topbot[1])
    
    y = np.linspace(1/x.shape[-1], 1, x.shape[-1], endpoint=True) # 1/(w*h) ... (w*h)/(w*h)
    
    return x, y