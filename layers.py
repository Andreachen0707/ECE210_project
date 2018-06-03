import numpy as np
import pdb


def affine_forward(x, w, b):
    dimension = 1
    x_cal = x
    for i in range(1, len(x.shape)):
        dimension *= x.shape[i]
    x_cal = x_cal.reshape(x.shape[0], dimension)
    out = np.dot(x_cal, w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None

    dimension = 1
    x_cal = x
    for i in range(1, len(x.shape)):
        dimension *= x.shape[i]
    x_cal = x_cal.reshape(x.shape[0], dimension)

    dx = np.dot(dout, w.T).reshape(x.shape)
    db = np.sum(dout, axis=0)
    dw = np.dot(x_cal.T, dout)

    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache

    out = x
    out[out > 0] = 1
    out[out < 0] = 0
    dx = dout * out

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        x_mu = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)

        running_mean = momentum * running_mean + (1 - momentum) * x_mu
        running_var = momentum * running_var + (1 - momentum) * x_var

        x_norm = (x - x_mu) / np.sqrt(x_var + eps)
        out = gamma * x_norm + beta
        cache = (x, x_norm, x_mu, x_var, gamma, beta, eps)

    elif mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

        cache = None

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None

    x, x_norm, x_mu, x_var, gamma, beta, eps = cache
    N, D = x.shape
    new_mu = x - x_mu

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma

    divar = np.sum(dx_norm * new_mu, axis=0)
    dxmu1 = dx_norm / np.sqrt(x_var + eps)

    dsqrtvar = -1. / (x_var + eps) * divar
    dvar = 0.5 * 1. / np.sqrt(x_var + eps) * dsqrtvar

    dsq = 1. / N * np.ones((N, D)) * dvar

    dxmu2 = 2 * new_mu * dsq

    dx1 = dxmu1 + dxmu2
    dmu = -1 * np.sum(dx1, axis=0)
    dx2 = 1. / N * np.ones((N, D)) * dmu

    dx = dx1 + dx2

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':


        mask = (np.random.rand(*x.shape) > p) / p
        # mask = np.random.rand(*x.shape)>p
        # mask = np.random.binomial(1,p,size = x.shape)
        out = x * mask


    elif mode == 'test':
        mask = np.ones(x.shape)
        out = x * mask

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):

    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':

        dx = mask * dout

    elif mode == 'test':

        dx = dout

    return dx


def svm_loss(x, y):

    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
