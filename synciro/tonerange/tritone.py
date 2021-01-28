import numpy as np

# simple separation
# TODO apply gamma for perceptive brightness / gray mode
# TODO grayscale mode to module
# TODO overlap options, blur, mask or applied

RGB2L_methods = {
            'L': [0.299, 0.587, 0.114], # ITU-R 601-2 Luma
            'Avg': [1/3, 1/3, 1/3],
            'L2': [0.2126, 0.7152, 0.0722],
        }

# remain for backward compatibility
"""
    Three-way tone range : Shadows, Midtones, Highlights
    Assuming last dimension of in_array is the channel.
"""
def split_tritone(in_array, midrange=(70,180), channel_coef='L', c=1):
    n_channel = in_array.shape[-1]
    _in_array = in_array.reshape(-1, n_channel)

    if n_channel == 1:
        img_L = _in_array[:,0]
    elif n_channel == 3:
        lc = RGB2L_methods[channel_coef]
        img_L = (lc[0]*(_in_array[:,0]**c) + 
                 lc[1]*(_in_array[:,1]**c) + 
                 lc[2]*(_in_array[:,2]**c))**(1/c)
    else:
        raise ValueError("image channel must be either 1 or 3")

    p,q = midrange

    shadow_mask = img_L < p
    midtones_mask = (img_L >= p) * (img_L < q)
    highlight_mask = img_L >= q

    return  shadow_mask.reshape(in_array.shape[:-1]), \
            midtones_mask.reshape(in_array.shape[:-1]), \
            highlight_mask.reshape(in_array.shape[:-1])


"""
    Split each last dimension of in_array into three by the midtones
    if supplied midrange is 1d, then it is broadcasted.
"""
def _split(in_array, midrange=((70,180))):
    n_channel = in_array.shape[-1]
    
    midrange = np.asarray(midrange)
    # when input is one pair, assume it is array of pair with 1 length.
    if len(midrange.shape) == 1: midrange = midrange[None,:]
    
    lo = in_array < midrange[:,0]
    md = (in_array >= midrange[:,0]) * (in_array < midrange[:,1])
    hi = in_array >= midrange[:,1]

    return lo, md, hi
    
