from config import *
from old_main import myPlotScatter
import matplotlib.pyplot as plt
import numpy as np
import random

low = -1*DATA_SAMPLE_RANGE
high = DATA_SAMPLE_RANGE
margin = (high-low)/20.

'''
    flag = 0 to assign to upper right
    flag = 1 to assign to lower left
'''
def seperateData(x1, x2, flag): 
    a = []
    b = []
    slop = 3
    
    if(flag == 0):
        for i in range(len(x1)):
            if x2[i] > slop*x1[i] + margin:
                a.append(x1[i])
                b.append(x2[i])
        return a, b
    else:
        for i in range(len(x1)):
            if x2[i] < slop*x1[i] - margin:
                a.append(x1[i])
                b.append(x2[i])
        return a, b
    
def addNoise(data, rou):
    arr = data[:]
    noise_index = random.sample(range(0, len(arr)-1), int(rou*len(arr)))
     
    for ix in range(len(noise_index)):
        if arr[noise_index[ix]][2] == -1:
            arr[noise_index[ix]][2] = 1
        else:
            arr[noise_index[ix]][2] = -1
    return arr

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


def generatData(_low, _high, _num):    
    '''
    Use Uniform distribution create num samples 
    '''
    x_1 = np.random.uniform(low = _low, high = _high, size = (_num,))
    x_2 = np.random.uniform(low = _low, high = _high, size = (_num,))
    x_3 = np.random.uniform(low = _low, high = _high, size = (_num,))
    x_4 = np.random.uniform(low = _low, high = _high, size = (_num,))

    '''
    Seperate 4 data array into two groups, one is upperright dataset 
    and another is lowerleft dataset. This is clean dataset.
    '''
    group_1, group_2 = seperateData(x_1, x_2, 0)
    label_data_1 = [1]*len(group_1)
    data_1 = np.vstack((group_1,group_2,label_data_1)).T

    group_3, group_4 = seperateData(x_3, x_4, 1)
    label_data_2 = [-1]*len(group_3)
    data_2 = np.vstack((group_3,group_4,label_data_2)).T
    
    dataset = np.vstack((data_1, data_2))

    return dataset
    
def saveToFile(file_name, dataArray):
    _file = open(file_name, 'w')
    for ix in range(len(dataArray)):
        for _item in dataArray[ix]:
            _file.write("%s\t" %_item)
        _file.write("\n")

if __name__ == "__main__":

    for ix in range(len(SAMPLE_ITEM_NUM)):
        data = generatData(low, high, SAMPLE_ITEM_NUM[ix])
        print data
        myPlotScatter(data)
        plt.show()
        file_name = 'clean_' + str(SAMPLE_ITEM_NUM[ix]) + '.txt'
        saveToFile(file_name, data)
