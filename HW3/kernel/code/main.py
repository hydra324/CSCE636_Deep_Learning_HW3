from DataReader import prepare_data
from model import Model

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def main():
    # ------------Data Preprocessing------------
    train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y = prepare_data(data_dir, train_filename, test_filename)
    print('trainX.shape: ',train_X.shape,'trainY.shape: ',train_y.shape)

    # ------------Kernel Logistic Regression Case------------
    ### YOUR CODE HERE
    # Run your kernel logistic regression model here
    learning_rate = 0.01
    max_epoch = 50
    batch_size = 128
    sigmas = [0.1,0.5,1.,3.,10.,30.]
    sigma = 3.0

    for sigma in sigmas:
        print(f'running with sigma={sigma}')
        model = Model('Kernel_LR', train_X.shape[0], sigma)
        model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model = Model('Kernel_LR', train_valid_X.shape[0], sigma)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE

    # ------------RBF Network Case------------
    ### YOUR CODE HERE
    # Run your radial basis function network model here
    
    learning_rate = 0.01
    max_epoch = 50
    batch_size = 128
    sigmas = [0.1,0.5,1.,3.,10.,30.]
    hidden_dims = [32,64,128]
    sigma = 3.0
    hidden_dim = 64

    for hidden_dim in hidden_dims:
        for sigma in sigmas:
            print(f'running with hidden_dim={hidden_dim} and sigma={sigma}')
            model = Model('RBF', hidden_dim, sigma)
            model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model = Model('RBF', hidden_dim, sigma)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE

    # ------------Feed-Forward Network Case------------
    ### YOUR CODE HERE
    # Run your feed-forward network model here
    learning_rate = 0.01 
    max_epoch = 50
    batch_size = 128
    hidden_dims = [32,64,128,256,512]
    hidden_dim = 64

    for hidden_dim in hidden_dims:
        print(f'running with hidden_dim={hidden_dim}')
        model = Model('FFN', hidden_dim)
        model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model = Model('FFN', hidden_dim)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("score = {} in test set\n".format(score))
    ### END YOUR CODE
    
if __name__ == '__main__':
    main()