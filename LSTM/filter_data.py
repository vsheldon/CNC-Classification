from sklearn import preprocessing
import math
import numpy as np


# This function preprocess raw data and store them into dictionary
def preprocess_csv(file):
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

def preprocess_txt():
    X1 = np.loadtxt('X1.txt')
    Y1 = np.loadtxt('Y1.txt')
    Z1 = np.loadtxt('Z1.txt')
    M1 = np.loadtxt('M1.txt')
    S1 = np.loadtxt('S1.txt')
    min_len = min(len(X1),min(len(Y1), min(len(Z1), min(len(M1), len(S1)))))
    data = np.concatenate((X1[:min_len,1:],Y1[:min_len,1:],Z1[:min_len,1:],M1[:min_len,1:],S1[:min_len,1:]), axis=1)
    return data

#this function filtered out the feature contains the same information
def filter(data): 
  row, col = data.shape
  valid_col = []
  for i in range(0, col):
    for j in range(1, row):
      if data[j][i] != data[j-1][i]:
        valid_col.append(i)
        break
  return valid_col


# this function reduces the feature dimensions
def format(data, valid_col):
  new_data = []
  row, col = data.shape
  for i in range(0, col):
    if i in valid_col:
      if new_data == []:
        new_data = data[:,i].reshape(-1,1)
      else:
        new_data = np.concatenate((new_data, data[:,i].reshape(-1,1)), axis=1)
  new_data = np.asarray(new_data)
  return new_data


def dimension_reduce(data, valid_col):
  percent_to_keep = 1.0 # change this value to determine the percentage features to keep
  
  cnt = math.ceil(float(len(valid_col))*percent_to_keep)

  while cnt != len(valid_col):
    max_cor = -2147483627
    del_idx = valid_col[0] 
    for i in valid_col:
      for j in valid_col:
        if i != j:
          a = data[:,i]
          b = data[:,j]
          if np.correlate(a, b) > max_cor:
            max_cor = np.correlate(a, b)
            del_idx = i
    valid_col.remove(del_idx)
  return valid_col


