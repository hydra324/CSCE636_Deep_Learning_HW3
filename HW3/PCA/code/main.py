import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 64, 300)
    # for batch_size in [32,64,128]:
    # model.train(A, A, batch_size, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    # final_w = None
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    # print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error, f' batch size={batch_size}')
    return final_w

if __name__ == '__main__':
    dataloc = "../data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
    # ps = [50, 100, 150]
    ps = [32,64,128,256]
    # ps = [64]
    for p in ps:
        G = test_pca(A, p)
        final_w = test_ae(A, p)
        difference = frobeniu_norm_error(G,final_w)
        print(f'difference between G and final_w for {p} dimensional representation is: ',difference)
        difference = frobeniu_norm_error(G.T @ G, final_w.T @ final_w)
        print(f'difference between G^TG and W^TW for {p} dimensional representation is: ',difference)
        R = G.T @ final_w
        print('G.shape: ',G.shape, 'W.shape: ',final_w.shape)
        _,n = G.shape
        difference = frobeniu_norm_error(R @ R.T,np.identity(n))
        print(f'difference between RR^T and I for {p} dimensional representation is: ',difference)
        difference = frobeniu_norm_error(R.T @ R,np.identity(n))
        print(f'difference between R^TR and I for {p} dimensional representation is: ',difference)
    ### END YOUR CODE 
