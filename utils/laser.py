import numpy as np

@np.vectorize
def linear_rampdown(t, t1=0.005, t2=0.010):
    """Implements linear rampdown laser pulse shape."""
 
    if t < t1:
        return 1.
    elif t < t2:
        return (t2-t)/(t2-t1)
    else:
        return 0.