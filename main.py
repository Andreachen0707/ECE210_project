import pandas as pd
import numpy as np
import network
import optims
from solver import *
import gzip

import matplotlib.pyplot as plt

def data_load (percentage = 0.8) :
    file_names = [['X_train', 'train-images-idx3-ubyte.gz'],
				 ['y_train', 'train-labels-idx1-ubyte.gz'],
				 ['X_test', 't10k-images-idx3-ubyte.gz'],
				 ['y_test', 't10k-labels-idx1-ubyte.gz']]

    mnist_data = {}
    for pair in file_names :
        with gzip.open(pair[1], 'rb') as f:
            if pair[0].startswith('X') :	# img
                mnist_data[pair[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
            else :	# label
                mnist_data[pair[0]] = np.frombuffer(f.read(), np.uint8, offset=8)


    idx = np.arange(mnist_data['X_train'].shape[0])
    np.random.shuffle(idx)
    length = idx.shape[0]


    train_idx = idx[:int(length*percentage)]
    val_idx = idx[int(length*percentage):]


    mnist_data['X_val'] = mnist_data['X_train'][val_idx]
    mnist_data['y_val'] = mnist_data['y_train'][val_idx]
    mnist_data['X_train'] = mnist_data['X_train'][train_idx]
    mnist_data['y_train'] = mnist_data['y_train'][train_idx]

    # print(mnist_data['y_train'])
    return mnist_data



def main():
    data = data_load()
    print('data_over')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    classes = 10


    lrs = [2e-4]


    for lr in lrs:
        # model = network.LogisticRegression(input_dim = X_train.shape[1],reg= 0.5, reg_type= 'l2')
        model = network.SVM(input_dim=X_train.shape[1])
        solver = Solver(model,data,num_epochs=40,batch_size=200,update_rule='sgd',
                        optim_config={'learning_rate':lr},
                        verbose=True,
                        print_every= 20
                        )
        solver.train()

        y_prob = model.loss(X_test)
        y_prob = np.argmax(y_prob, axis=1)

        test_acc = np.mean(y_prob == y_test)
        print('test_acc = ', test_acc)

        plt.plot(solver.loss_history)
        plt.hold(True)
    plt.show()



    # y_probs = []
    # for class_now in range(classes):
    #     y_train_ = one_hot(y_train,class_now)
    #     y_test_ = one_hot(y_test,class_now)
    #     # training
    #     model.train(X_train,y_train_,num_iter=1000,batch_size=500,lr=9e-6,verbose=True)
    #     print('classifier: {}'.format(class_now))
    #     # testing
    #     y_prob = model.predict(X_test)
    #     y_probs.append(y_prob)
    #
    # y_probs = np.argmax(y_probs,axis = 0)
    # y_test = y_test.reshape(-1,1)
    # test_acc = np.sum(y_probs==y_test)/y_test.shape[0]
    # print(test_acc)

    # model = network.SVM(dims=(classes,X_train.shape[1]))
    # model.train(X_train,y_train,num_iters=2000,batch_size=500,learning_rate=2e-3,verbose=True)
    # y_pred = model.predict(X_test)
    # print(y_pred[:10],y_test[:10])
    # test_acc = np.sum(y_pred==y_test)/y_test.shape[0]
    # print(test_acc)







if __name__ == '__main__':
    main()



