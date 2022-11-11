import torch
import torch.nn as nn
import numpy as np

"""
This script implements a kernel logistic regression model, a radial basis function network model
and a two-layer feed forward network.
"""

class Kernel_Layer(nn.Module):

    def __init__(self, sigma, hidden_dim=None):
        """
        Set hyper-parameters.
        Args:
            sigma: the sigma for Gaussian kernel (radial basis function)
            hidden_dim: the number of "kernel units", default is None, then the number of "kernel units"
                                       will be set to be the number of training samples
        """
        super(Kernel_Layer, self).__init__()
        self.sigma = sigma
        self.hidden_dim = hidden_dim
    
    def reset_parameters(self, X):
        """
        Set prototypes (stored training samples or "representatives" of training samples) of
        the kernel layer.
        """
        if self.hidden_dim is not None:
            X = self._k_means(X)
        self.prototypes = nn.Parameter(torch.tensor(X).float(), requires_grad=False)
    
    def _k_means(self, X):
        """
        K-means clustering
        
        Args:
            X: A Numpy array of shape [n_samples, n_features].
        
        Returns:
            centroids: A Numpy array of shape [self.hidden_dim, n_features].
        """
        ### YOUR CODE HERE
        # self.hidden_dim -> number of centroids

        # step-1 initialize random centroids
        (n_samples,n_features) = X.shape
        random_centroid_indices = np.random.randint(0,n_samples,size=self.hidden_dim)
        centroids = X[random_centroid_indices]

        it,max_iterations = 0,100
        is_converged = False
        while (not is_converged) and it < max_iterations:
        
            # step-2 compute distance to each centroid and assign
            clusters = [[] for i in range(self.hidden_dim)]
            for x in X:
                # find new index for x
                new_index = min([(idx, (x-centroid)@(x-centroid)) for idx,centroid in enumerate(centroids)],key=lambda p: p[1])[0]
                # assign to cluster for next centroid computation
                clusters[new_index].append(x)
            
            # step-3 compute new centroids
            new_centroids = [np.mean(np.array(cluster),axis=0) for cluster in clusters]

            # step-3.1 check for any empty centroids
            for idx,centroid in enumerate(new_centroids):
                if np.isnan(centroid).any():
                    # assign prev centroid
                    new_centroids[idx]=centroids[idx]

            # step-4 check for convergence
            is_converged = np.allclose(np.sort(centroids, axis=1), np.sort(new_centroids, axis=1))
            centroids = new_centroids
            it += 1
        print(f'k-means algorithm converged after {it} iterations')
        ### END YOUR CODE
        return centroids
    
    def forward(self, x):
        """
        Compute Gaussian kernel (radial basis function) of the input sample batch
        and self.prototypes (stored training samples or "representatives" of training samples).

        Args:
            x: A torch tensor of shape [batch_size, n_features]
        
        Returns:
            A torch tensor of shape [batch_size, num_of_prototypes]
        """
        assert x.shape[1] == self.prototypes.shape[1]
        ### YOUR CODE HERE
        # Basically you need to follow the equation of radial basis function
        # in the section 5 of note at http://people.tamu.edu/~sji/classes/nnkernel.pdf

        # self.prototypes seems to have the centroids that were computed using K-means clustering
        # instead of writing in loops, let's utilize broadcasting in pytorch
        # shape of x = (B,d) #(batchsize,features)
        # shape of prototypes or centroids = (c,d) #(number of centroids,features)
        # output shape we want = (B,c) #(batchsize,number of centroids)

        prototypes = torch.unsqueeze(self.prototypes, 0) # new shape: (1,c,d)
        x = torch.unsqueeze(x, 1) # new shape: (B,1,c)
        # note that python broadcast will take care of repeating along dimensions in prototypes-x
        return torch.exp(-1*((x-prototypes) ** 2).sum(-1)/(2*self.sigma*self.sigma))
        ### END YOUR CODE


class Kernel_LR(nn.Module):

    def __init__(self, sigma, hidden_dim):
        """
        Define network structure.

        Args:
            sigma: used in the kernel layer.
            hidden_dim: the number of prototypes in the kernel layer,
                                       in this model, hidden_dim has to be equal to the 
                                       number of training samples.
        """
        super(Kernel_LR, self).__init__()
        self.hidden_dim = hidden_dim
        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of a
        # kernel layer (Kernel_Layer object) and a linear layer (nn.Linear object)

        # Remember that kernel logistic regression model uses all training samples
        # in kernel layer, so set 'hidden_dim' argument to be None when creating
        # a Kernel_Layer object.

        # How should we set the "bias" argument of nn.Linear? 
        self.net = torch.nn.Sequential(
            Kernel_Layer(sigma=sigma,hidden_dim=None),
            torch.nn.Linear(self.hidden_dim,1,bias=False),
        )
        
        ### END YOUR CODE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]
        
        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)
    
    def reset_parameters(self, X):
        """
        Initialize the weights of the linear layer and the prototypes of the kernel layer.

        Args:
            X: A Numpy array of shape [n_samples, n_features], training data matrix.
        """
        assert X.shape[0] == self.hidden_dim
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                if isinstance(layer, Kernel_Layer):
                    layer.reset_parameters(X)
                else:
                    layer.reset_parameters()



class RBF(nn.Module):

    def __init__(self, sigma, hidden_dim):
        """
        Define network structure.

        Args:
            sigma: used in the kernel layer.
            hidden_dim: the number of prototypes in the kernel layer,
                                       in this model, hidden_dim is a user-specified hyper-parameter.
        """
        super(RBF, self).__init__()
        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of a
        # kernel layer (Kernel_Layer object) and a linear layer (nn.Linear object)
        # How should we set the "bias" argument of nn.Linear? 

        # Since kernel logistic regression doesn't have bias, we set bias as False in the dense layer
        self.net = torch.nn.Sequential(
            Kernel_Layer(sigma=sigma,hidden_dim=hidden_dim),
            torch.nn.Linear(hidden_dim,1,bias=False),
        )
        
        ### END CODE HERE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]
        
        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)
    
    def reset_parameters(self, X):
        """
        Initialize the weights of the linear layer and the prototypes of the kernel layer.

        Args:
            X: A Numpy array of shape [n_samples, n_features], training data matrix.
        """
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                if isinstance(layer, Kernel_Layer):
                    layer.reset_parameters(X)
                else:
                    layer.reset_parameters()



class FFN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        """
        Define network structure.

        Args:
            input_dim: number of features of each input.
            hidden_dim: the number of hidden units in the hidden layer, a user-specified hyper-parameter.
        """
        super(FFN, self).__init__()
        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of
        # two linear layers (nn.Linear object)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,1)
        )
        ### END CODE HERE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]
        
        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)

    def reset_parameters(self):
        """
        Initialize the weights of the linear layers.
        """
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()