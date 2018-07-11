from distutils.text_file import TextFile
from config import *
import matplotlib.pyplot as plt
import numpy as np
import random


def subArrayColumn(data):
    size = len(data)
    a = [0]*size
    b = [0]*size
    c = [0]*size
    
    for ix in range(size):
        a[ix] = data[ix][0]
        b[ix] = data[ix][1]
        c[ix] = data[ix][2]
        
    return a,b,c

def getTwoDatasetForPlot(data):
    size = len(data)
    result_positive = []
    result_negative = []
    
    for ix in range(size):
        if data[ix][2] == 1:
            result_positive.append(data[ix])
        else:
            result_negative.append(data[ix])
    
    return result_positive, result_negative

def addNoiseDiffRou(data, rou_plus, rou_minus):
    arr = data[:]
    arr_plus = []
    arr_minus = []
    
    for a in arr:
        if a[2] == 1:
            arr_plus.append(a) 
        else:
            arr_minus.append(a)
            
    noise_index_plus = random.sample(range(0, len(arr_plus)-1), int(rou_plus*len(arr_plus)))
    for ix in range(len(noise_index_plus)):
        arr_plus[ix][2] = -1
                  
    noise_index_minus = random.sample(range(0, len(arr_minus)-1), int(rou_minus*len(arr_minus)))
    for ix in range(len(noise_index_minus)):
        arr_minus[ix][2] = 1    
            
    arr = np.vstack((arr_plus, arr_minus))
    
    return arr

def myPlotScatter(data):
    positive_arr, negative_arr = getTwoDatasetForPlot(data)

    graph_positive_x1, graph_positive_x2, positive_label = subArrayColumn(positive_arr)
    graph_negative_x1, graph_negative_x2, negative_label = subArrayColumn(negative_arr)
    
    plt.scatter(graph_positive_x1,graph_positive_x2, marker='o', c='b', s=10, label='x_2')
    plt.scatter(graph_negative_x1,graph_negative_x2, marker='x', c='r', s=10, label='x_1')
    
    plt.title('Dataset')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
        
def readTextFile(file_name):
    t_file = open(file_name)
    lines = t_file.readlines()
    result = []
    for line in lines:
        _temp = []
        for _item in line.split():
            _temp.append(float(_item))
        result.append(_temp)
    return result
    
    

if __name__ == "__main__":
    data = readTextFile('noise_output_data_100_samples_and_rou_0.4.txt')
    myPlotScatter(data)
    plt.show()
