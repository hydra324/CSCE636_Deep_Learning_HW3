import torch
import torch.nn as nn
import numpy as np
import tqdm
from network import Kernel_LR, RBF, FFN


class Model(object):
    
    def __init__(self, network, hidden_dim, sigma=0):
        """
        Define model object.
        """
        assert network in ['Kernel_LR', 'RBF', 'FFN']
        if network == 'Kernel_LR':
            self.model = Kernel_LR(sigma, hidden_dim)
        elif network == 'RBF':
            self.model = RBF(sigma, hidden_dim)
        else:
            self.model = FFN(256, hidden_dim)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X: A Numpy array of shape [n_samples, n_features].

        Returns:
            preds: A Numpy array of shape [n_samples,]. Only contains 1 or 0.
        """
        X_tensor = torch.tensor(X).float()
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            self.model = self.model.cuda()
        
        self.model.eval()
        with torch.no_grad():
            z = self.model(X_tensor).cpu().numpy()
            preds = (z >= 0).reshape(-1).astype(np.int32)
        
        return preds

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X: A Numpy array of shape [n_samples, n_features].
            y: A Numpy array of shape [n_samples,]. Only contains 1 or 0.

        Returns:
            score: A float. Mean accuracy of self.predict(X) wrt. y.
        """
        preds = self.predict(X)
        is_correct = (y == preds).astype(np.float64)
        score = np.sum(is_correct) / is_correct.size
        return score
    
    def train(self, train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size):
        """
        Train model on data (train_X, train_y) with batch gradient descent.

        Args:
            train_X, train_y: Arrays of shape [n_train_samples, 256] and [n_train_samples,], data and labels of training set.
            valid_X, valid_y: Arrays of shape [n_valid_samples, 256] and [n_valid_samples,], data and labels of validation set.
            max_epoch: Number of training epochs, a user-specified hyper-parameter.
            learning_rate: Learning rate, a user-specified learning rate.
            batch_size: Batch size, a user-specified batch size.
        """

        # initialize model parameters
        if isinstance(self.model, FFN):
            self.model.reset_parameters()
        else:
            self.model.reset_parameters(train_X)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) # initialize optimizer
        criterion = nn.BCEWithLogitsLoss(reduction='mean') # initialize loss function
        num_batches = int(np.ceil(train_X.shape[0] / batch_size))
        
        for _ in range(max_epoch):
            idxs = np.arange(train_X.shape[0])
            np.random.shuffle(idxs)

            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                idx = idxs [i : min(i+batch_size, train_X.shape[0])]            
                X_batch, y_batch = train_X[idx], train_y[idx]
                X_batch_tensor = torch.tensor(X_batch).float()
                y_batch_tensor = torch.tensor(y_batch).float()
                if torch.cuda.is_available():
                    X_batch_tensor = X_batch_tensor.cuda()
                    y_batch_tensor = y_batch_tensor.cuda()
                    self.model = self.model.cuda()

                y_pred = self.model(X_batch_tensor)
                loss = criterion(y_pred, y_batch_tensor.view(-1,1))
                
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # do validation
            if valid_X is not None and valid_y is not None:
                score = self.score(valid_X, valid_y)
                print("score = {} in validation set.\n".format(score))
            
            # set the network to train mode again
            self.model.train()
            