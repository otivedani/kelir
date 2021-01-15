import numpy as np

# simple separation
# TODO apply gamma for perceptive brightness / gray mode
# TODO grayscale mode to module
# TODO overlap options, blur, mask or applied

def threeway(img, midrange=(70,180), channel_coef='L', c=1):
    
    lmode = {
        'L': [0.299, 0.587, 0.114], # ITU-R 601-2 Luma
        'Avg': [1/3, 1/3, 1/3],
        'L2': [0.2126, 0.7152, 0.0722],
    }[channel_coef]

    img_L = (lmode[0]*img[:,:,0]**c + lmode[1]*img[:,:,1]**c + lmode[2]*img[:,:,2]**c)**(1/c)

    p,q = midrange

    shadow_mask = img_L < p
    midtones_mask = (img_L >= p) * (img_L < q)
    highlight_mask = img_L >= q

    return shadow_mask, midtones_mask, highlight_mask