from __future__ import division
import numpy as np
import optims


class LogisticRegression(object):
    def __init__(self, input_dim, reg = 0.1, classes = 10, reg_type = 'None'):
        '''
        constructor just takes number of iterations for gradient descent and value of alpha.
        '''
        self.params = {}
        self.classes = classes
        self.reg = reg
        self.reg_type = reg_type
        self.input_dim = input_dim  # 784 image size shape[1]
        # self.params['W'] = np.random.normal(size= [self.input_dim,self.classes])
        self.params['W'] = np.zeros((self.input_dim,self.classes))

    def one_hot(self, y, class_now):
        y_ = np.zeros(y.shape)
        y_[y == class_now] = 1
        return y_



    def loss_single(self,X,y,W):
        reg_num = 0
        grad_num = 0
        if self.reg_type is 'l1':
            reg_num = self.reg* np.sum( np.abs(W))
            grad_num = self.reg*(np.sign(W))
        elif self.reg_type is 'l2':
            reg_num = 0.5 * self.reg*np.sum(W*W)
            grad_num = self.reg * W
        m, n = X.shape
        y = y.reshape(-1,1)
        h = 1/(1+np.exp(-np.dot(X,W)))
        J = (-1.0/m)*(np.sum(np.log(h)*y+(np.log(1-h) * (1 - y)))) + reg_num

        grad = 1/m*np.dot(X.T,(h-y)) + grad_num
        return J, grad

    def loss(self,X,y = None):
        loss = []
        grad = {}
        grad_temp = []

        if y is None:
            return 1/(1+np.exp(-np.dot(X,self.params['W'])))

        for class_id in range(self.classes):
            y_onehot = self.one_hot(y,class_id)
            W_single = self.params['W'][:,class_id]
            W_single = W_single.reshape(-1,1)
            # print(W_single.shape)
            loss_, grad_ = self.loss_single(X,y_onehot,W_single)
            grad_temp.append(grad_)
            loss.append(loss_)

        grad_temp = np.asarray(grad_temp)[:,:,0]
        grad['W'] = np.array(grad_temp.T)
        return np.mean(loss),grad


class SVM(object):
    def __init__(self, input_dim, classes = 10, reg = 0.0):
        self.params = {}
        self.dims = input_dim
        self.classes = classes
        self.reg = reg
        self.params['W'] = np.random.normal(size=[self.classes, self.dims])

    def loss(self, X, y=None):
        """
        A vectorized implementation of loss_and_grad. It shares the same
    	inputs and ouptuts as loss_and_grad.
        """
        loss = 0.0
        grad = {}
        grad['W'] = np.zeros(self.params['W'].shape)  # initialize the gradient as zero
        if y is None:
            return np.dot(X, self.params['W'].T)

        a_j = np.dot(X, self.params['W'].T)  # score
        a_yj = a_j[np.arange(X.shape[0]) , y]  # true values

        max_out = 1 + a_j - a_yj.reshape(X.shape[0], 1)  # svm expression
        max_out[np.arange(X.shape[0]), y] = 0  # remove right results where j equal to yi

        loss = np.sum(np.maximum(np.zeros((X.shape[0], self.params['W'].shape[0])), max_out))
        loss = loss / X.shape[0] + 0.5*self.reg*np.sum(self.params['W']**2)

        index = np.maximum(np.zeros((X.shape[0], self.params['W'].shape[0])), max_out)  # compare with 0
        index[index > 0] = 1  # set greater than 0 as 1
        max_index = np.sum(index, axis=1)  # counts of incorrect results
        index[np.arange(X.shape[0]), y] = -max_index[np.arange(X.shape[0])]  # svm expression

        grad['W'] = np.dot(index.T, X) / X.shape[0] + self.reg* self.params['W']

        return loss, grad


class SVM_kernel(object):
    def __init__(self, input_dim, classes = 10, reg = 0.1):
        self.params = {}
        self.dims = input_dim
        self.classes = classes
        self.reg = reg
        self.params['W'] = np.random.normal(size=[self.classes, 100])

    def rbf(self, x, y):
        gamma = 0.05
        return np.exp(-gamma * np.linalg.norm(x-y) ** 2)

    def loss(self, X, y=None):
        """
        A vectorized implementation of loss_and_grad. It shares the same
    	inputs and ouptuts as loss_and_grad.
        """
        loss = 0.0
        X_shape = X.shape[0]

        K = np.zeros((X_shape,X_shape))

        # print(X[0,:].shape)

        for i in range(X_shape):
            for j in range(X_shape):
                X_1 = X[i,:].reshape(-1,1)
                X_2 = X[j, :].reshape(-1, 1)
                K[i, j] = self.rbf(X_1,X_2)
        # print(K.shape)

        grad = {}
        grad['W'] = np.zeros(self.params['W'].shape)  # initialize the gradient as zero
        if y is None:
            return np.dot(K, self.params['W'].T)

        a_j = np.dot(K, self.params['W'].T)  # score
        a_yj = a_j[np.arange(K.shape[0]) , y]  # true values

        max_out = 1 + a_j - a_yj.reshape(K.shape[0], 1)  # svm expression
        max_out[np.arange(K.shape[0]), y] = 0  # remove right results where j equal to yi

        loss = np.sum(np.maximum(np.zeros((K.shape[0], self.params['W'].shape[0])), max_out))
        loss = loss / K.shape[0] # + 0.5 * self.reg * np.dot(np.dot(self.params['W'], K),self.params['W'].T)

        index = np.maximum(np.zeros((K.shape[0], self.params['W'].shape[0])), max_out)  # compare with 0
        index[index > 0] = 1  # set greater than 0 as 1
        max_index = np.sum(index, axis=1)  # counts of incorrect results
        index[np.arange(K.shape[0]), y] = -max_index[np.arange(K.shape[0])]  # svm expression

        grad['W'] = np.dot(index.T, K) / K.shape[0] #+ self.reg * np.dot(K, self.params['W'].T ).T

        return loss, grad




class FCNet(object):
    def __init__(self, input_dim, classes=10):
        self.params = {}
        self.dims = input_dim
        self.classes = classes
        self.params['W'] = np.random.normal(size=[self.classes, self.dims])

    def loss(self):
        pass





