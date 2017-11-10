from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from lstm import LstmParam, LstmNetwork


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def main():
	[train, label] = example0()
	example_0(train, label)


def example_0(x,y):
    # learns to repeat simple sequence from random inputs
    #np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 50
    x_dim = 2
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    #y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = x
    y_list = y

    #input_val_arr = [np.random.random(x_dim) for _ in y_list]
    for cur_iter in range(100):
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()



def data_preprocessing(file):
    data, tmp, pre, tmp2, data_raw = {}, [], '', [], {}
    for line in open(file,'r').readlines():
        line = line.split(',')
        if line[0] == 'name' and pre != '':
            tmp2 = tmp
            data_raw[pre] = np.asarray(tmp2,dtype='f')
            data[pre] = preprocessing.normalize(np.asarray(tmp2,dtype='f'))
            tmp = []
        elif line[0] == 'name': 
            data['name'] = line[2:]
            data_raw['name'] = line[2:]
        if line[2].isdigit() == True:
            row = []
            for i in range(2,47):
                row.append(0.0 if line[i]=='' else float(line[i]))	
            tmp.append(np.asarray(row, dtype='f'))
            pre = line[0]
    tmp2 = tmp
    data[pre] = preprocessing.normalize(np.asarray(tmp2,dtype='f'))
    data_raw[pre] = np.asarray(tmp2,dtype='f')
    return [data, data_raw]

def example0():
	train_pos = 'G_Fifty_9'
	train_neg = 'G_Hundred_1'
	test_pos = 'G_Fifty_2'
	test_neg = 'G_Hundred_4'
	data,raw = data_preprocessing('data.csv')
	# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
	plt.plot(data[train_pos][:,26])
	plt.ylabel('Y1.Actual Acceleration')
	plt.xlabel('data index')
	mean = np.mean(data[train_pos], axis=1)
	var = np.var(data[train_pos], axis=1)
	dict, cor = {}, {}

	for i in range(45):
	    for j in range(45):
	        if i != j:
	            dict[(i,j)] = abs((mean[i]-mean[j]))/(var[i]+var[j]) - np.correlate(data[train_pos][:,j], data[train_pos][:,i])
	            
	            
	# x1 = data[train_pos]
	# x2 = data[train_neg]
	# x1_ = data[test_pos]
	# x2_ = data[test_neg]
	x1 = raw[train_pos]
	x2 = raw[train_neg]
	# x1_ = raw[test_pos]
	# x2_ = raw[test_neg]
	X1 = np.concatenate((x1[:,4],x2[:,4]), axis=0).reshape(-1,1)
	X2 = np.concatenate((x1[:,26],x2[:,26]), axis=0).reshape(-1,1)
	X =  np.concatenate((X1, X2), axis=1)
	y = np.concatenate((0*np.ones(len(data[train_pos])),np.ones(len(data[train_neg]))),axis=0).reshape(-1,1)

	# X1_ = np.concatenate((data[test_pos][:,4],data[test_neg][:,4]), axis=0).reshape(-1,1)
	# X2_ = np.concatenate((data[test_pos][:,26],data[test_neg][:,26]), axis=0).reshape(-1,1)
	# X_ =  np.concatenate((X1_, X2_), axis=1)

	return [X, y]

if __name__ == '__main__':
	main()
	
	