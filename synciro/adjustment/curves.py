import numpy as np
from scipy import interpolate

"""
Apply curves to image

param : 
    - image : numpy array, with shape (width, height, 3)
    - points : list of each channel curve points (x,y) 
        ~ len(points) must be either : 
            - 1 (equal adjust to all channel)
            - 3 (respective adjust to each channel)
            - 4 (respective adjust each and all channel)
        ~ len(points[i]) keep it < 16 for best result
        ~ len(points[i][j]) is a pair of points [x,y]
"""
def adjust_curves(image, points):

    strat = len(points)
    assert(strat in { 1, 3, 4 })

    def _prep_pairs(p):
        # create and check
        pt = np.array(p, dtype=np.uint8)
        assert(pt.shape[-1]==2)
        # sort
        pt = pt[np.argsort(pt[:,0])]
        # if x first not 0, shift to right, add 0,0 and 255,255 pair
        if not np.any(pt[:,0] == 0): pt = np.vstack(([0,0], pt))
        if not np.any(pt[:,0] == 255): pt = np.vstack((pt, [255,255]))
        
        return pt

    # points to interpolate
    pt_vectors = [_prep_pairs(p) for p in points]

    # interpolation functions
    curves_f = [ interpolate.interp1d(pt_v[:,0], pt_v[:,1], kind='quadratic') for pt_v in pt_vectors ]

    # prepare output
    out_image=image.copy()

    # case 3 : apply to each channel separately (c0, c1, c2)->(R, G, B)
    # case 1 : apply to all channel equally (c0)->(RGB)
    # case 4 : apply 3 and 1 (c1,c2,c3)->(R, G, B) ; (c0)->(RGB)
    _apply_each = strat in {3,4}
    _apply_all = strat in {1,4}
    _flag = int(_apply_all)
    
    if (_apply_each):
        for i, curve_f in enumerate(curves_f[_flag:]):
            out_image[:,:,i] = np.clip(curve_f(out_image[:,:,i]), 0, 255)

    if (_apply_all):
        out_image = np.clip(curves_f[0](out_image), 0, 255)

    # applied curves
    return out_image.astype(image.dtype)



    