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


def adam(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('a', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = w
    config['t'] += 1
    config['v'] = config['beta1'] * config['v'] + (1 - config['beta1']) * dw
    config['a'] = config['beta2'] * config['a'] + (1 - config['beta2']) * dw ** 2

    m_hat = config['v'] / (1 - config['beta1'] ** config['t'])
    v_hat = config['a'] / (1 - config['beta2'] ** config['t'])
    next_w = next_w - config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])

    return next_w, config