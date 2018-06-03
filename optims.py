import numpy as np


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)  # set momentum to 0.9 if it wasn't there
    v = config.get('velocity', np.zeros_like(w))  # gets velocity, else sets it to zero.

    next_w = None

    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v

    config['velocity'] = v

    return next_w, config