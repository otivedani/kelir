import numpy as np

# simple separation
# TODO apply gamma for perceptive brightness / gray mode
# TODO overlap options, blur, mask or applied

def threeway(img, midrange=(70,180)):
    # ITU-R 601-2 Luma
    img_L = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

    p,q = midrange

    shadow_mask = img_L < p
    midtones_mask = (img_L >= p) * (img_L < q)
    highlight_mask = img_L >= q

    return shadow_mask, midtones_mask, highlight_mask