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

def pulse(control_abs, timeline):
    '''Laser pulse as numpy array.

    Parameters:
        control_abs (array-like (Nt,), float):
            The control function in absolute power scale (typically control * P_YAG).
        timeline (array-like (Nt,), float):
            Discrete time domain.

    '''

    return np.column_stack((control_abs, timeline))
