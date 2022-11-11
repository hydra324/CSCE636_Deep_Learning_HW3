import numpy as np 
import math
import matplotlib.pyplot as plt
import scipy.io as scio

def load_data(dataloc):
	data = scio.loadmat(dataloc)
	return data['A']

