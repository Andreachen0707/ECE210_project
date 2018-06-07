from __future__ import division
import numpy as np
import random
import optims

# can be put in solver
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

#can be put in solver
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


#single, not in sovler
class SVM_Kernel(object):
    def __init__(self, max_ite = 100, kernel_name = 'linear', C = 5, batch_size = 200, gamma_in = 0.05):
        self.C = C
        self.batch_size = batch_size
        self.gamma_in = gamma_in
        self.kernel = kernel_name
        self.ite = max_ite

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def rbf(self,X1,X2):
        self.gamma_in = 0.05
        n = X1.shape[0]
        m = X2.shape[0]
        num_1 = np.sum(X1**2,axis=1)
        num_2 = np.sum(X2**2,axis=1)
        X1_sum = np.tile(num_1,(m,1)).T
        X2_sum = np.tile(num_2,(n,1))

        # print(X1_sum.shape, X2_sum.shape)

        return np.exp(-self.gamma_in * (X1_sum + X2_sum -2* np.dot(X1,X2.T)))

    def train(self, X, Y):
        # input X size all dataset
        X_count = X.shape[0]
        num_batch = X_count / self.batch_size

        # Variables
        alphas = np.zeros(X_count)
        b = 0
        eta = 0
        L = 0
        H = 0
        tol = 1e-3


        for ite in range(self.ite):
            if ite%50 ==0:
                print(ite)
            # take batch
            for batch in range(int(num_batch)):
                # print('epoch {}:'.format(batch))
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                X_batch = X[start:end]
                Y_batch = Y[start:end]
                alpha_batch = alphas[start:end]
                if self.kernel is 'linear':
                    K = self.linear_kernel(X_batch,X_batch)
                else:
                    K = self.rbf(X_batch,X_batch)
                # print(K.shape)

                E = np.zeros(self.batch_size)
                for i in range(self.batch_size):
                    E[i] = b + sum(np.multiply(np.multiply(alpha_batch, Y_batch), K[:, i])) - Y_batch[i]
                    # print(np.dot(alpha_batch, Y_batch))

                    if ((Y_batch[i] * E[i] < -tol and alpha_batch[i] < self.C) or (
                            Y_batch[i] * E[i] > tol and alpha_batch[i] > 0)):
                        # random select j
                        j = round(self.batch_size * random.random()) - 1
                        if j == i:
                            j = round(self.batch_size * random.random()) - 1

                        # Calculate E(j)
                        E[j] = b + sum(np.multiply(np.multiply(alpha_batch, Y_batch), K[:, j])) - Y_batch[j]

                        # Save old alphas
                        alpha_i_old = alpha_batch[i]
                        alpha_j_old = alpha_batch[j]

                        # Compute L and H
                        if (Y_batch[i] == Y_batch[j]):
                            L = max(0, alpha_batch[j] + alpha_batch[i] - self.C)
                            H = min(self.C, alpha_batch[j] + alpha_batch[i])
                        else:
                            L = max(0, alpha_batch[j] - alpha_batch[i])
                            H = min(self.C, self.C + alpha_batch[j] - alpha_batch[i])

                        if L == H:
                            continue

                        # Compute eta
                        eta = 2 * K[i, j] - K[i, i] - K[j, j]
                        # print("eta",eta)
                        if eta >= 0:
                            continue

                        # Compute alpha
                        alpha_batch[j] = alpha_batch[j] - (Y_batch[j] * (E[i] - E[j])) / eta
                        alpha_batch[j] = min(H, alpha_batch[j])
                        alpha_batch[j] = max(L, alpha_batch[j])

                        if (abs(alpha_batch[j] - alpha_j_old) < tol):
                            alpha_batch[j] = alpha_j_old
                            continue

                        alpha_batch[i] = alpha_batch[i] + Y_batch[i] * Y_batch[j] * (alpha_j_old - alpha_batch[j])

                        # b
                        b1 = b - E[i]- Y_batch[i]*(alpha_batch[i]-alpha_i_old) * K[i,j].T \
                             - Y_batch[j]* (alpha_batch[j]-alpha_j_old) * K[i,j].T

                        b2 = b - E[j]- Y_batch[i]*(alpha_batch[i]-alpha_i_old) * K[i,j].T \
                             - Y_batch[j]* (alpha_batch[j]-alpha_j_old) * K[j,j].T

                        if (0 < alpha_batch[i] and alpha_batch[i]<self.C):
                            b = b1
                        elif (0 < alpha_batch[j] and alpha_batch[j]<self.C):
                            b = b2
                        else:
                            b = (b1+b2)/2
                        # print(alpha_batch.shape)

        return alphas,b

    def test(self,X_train,Y_train,X_test,Y_test,alpha,b):
        m = Y_test.size
        p = np.zeros(m)
        pred = np.zeros(m)

        alpha_idx = np.where(alpha > 0)

        X_train_small = X_train[alpha_idx]
        print(X_train_small.shape)
        Y_train_small = Y_train[alpha_idx]
        alpha_ = alpha[alpha_idx]

        if self.kernel is 'linear':
            W = np.dot(np.multiply(alpha_,Y_train_small).T, X_train_small)
            p = np.dot(X_test,W.T) + b

        else:
            K = self.rbf(X_test,X_train_small)
            # print(K.shape)
            p = K*Y_train_small.T
            p = p*alpha_.T
            p = np.sum(p,axis=1)

        return p



#can be put in solver
class FCNet(object):
    def __init__(self, input_dim, classes=10):
        self.params = {}
        self.dims = input_dim
        self.classes = classes
        self.params['W'] = np.random.normal(size=[self.classes, self.dims])

    def loss(self):
        pass

