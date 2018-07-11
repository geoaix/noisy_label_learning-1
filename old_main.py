from config import *
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from readFileToArray import readTextFile


initial_w = INITIAL_W
scaled_size = 1


# Here rou_plus and rou_minus choose the same value for the noise should happen on both classes
rou_plus = ROU_PLUS
rou_minus = ROU_MINUS
data_split_ratio = 0.7

plot_my_x_1 = []
plot_my_x_2 = []
plot_my_x_1.append(INITIAL_W[1])
plot_my_x_2.append(INITIAL_W[2])


'''
    add x_0 to x and set it to 1 and w_0 is b
'''
def getLinearFunctionProd(x, w):
    _x = []
    _x.append(1)
    _x.append(x[0])
    _x.append(x[1])
    return sum([v_1*v_2 for v_1, v_2 in zip(_x, w)])

def risk(data, w):
    _n = len(data)
    result = 0
    for _row in data:
        _y = _row[2]  
        _x = [1.]
        _x.append(_row[0])
        _x.append(_row[1])
        _row = _x
        y_time_w_time_x = _y * sum([a*b for a,b in zip(w, _row)])
        _ratio_1 = (1 - rou_minus)
        _ratio_2 = (rou_plus)
        row_1 = _ratio_1*math.log((1 + math.exp(-1*y_time_w_time_x)), math.e)
        row_2 = _ratio_2*math.log((1 + math.exp(y_time_w_time_x)), math.e)
        row_result = (row_1- row_2)/(math.log(2, math.e)*(1-rou_plus-rou_minus))
        result += row_result
    result = result/_n
    return result

def riskCountWithScaled(data, w):
    _n = len(data)
    result = 0
    for _row in data:
        _y = _row[2]        
        _x = [1.]
        _x.append(_row[0])
        _x.append(_row[1])
        _row = _x   
        y_time_w_time_x = _y * sum([a*b for a,b in zip(w, _row)])
        _ratio_1 = (1 - rou_minus)
        _ratio_2 = (rou_plus)
        row_1 = _ratio_1*math.log((1 + math.exp(-1*y_time_w_time_x)), math.e)
        row_2 = _ratio_2*math.log((1 + math.exp(y_time_w_time_x)), math.e)
        row_result = (row_1- row_2)/(math.log(2, math.e)*(1-rou_plus-rou_minus))
        result += row_result
    result /= (_n*scaled_size)
    return result

def getriskCountGradient(data, w):
    _n = len(data)
    result = []
    for _row in data:
        _y = _row[2]
        
        _x = [1.]
        _x.append(_row[0])
        _x.append(_row[1])
        _row = _x
        
        y_time_w_time_x = _y * sum([a*b for a,b in zip(w, _row)])
        _ratio_1 = ((1 - rou_minus)*(-1 * _y))/(math.exp(y_time_w_time_x) + 1)
        _ratio_2 = (rou_plus*_y)/(math.exp(-1*y_time_w_time_x) + 1)
        row_1 = [_ratio_1*a for a in _row]
        row_2 = [_ratio_2*a for a in _row]
        row_result_temp = [a-b for a,b in zip(row_1, row_2)]
        row_result = [a/(math.log(2, math.e)*(1-rou_plus-rou_minus)) for a in row_result_temp]
        result.append(row_result)
    
    w_0 = 0
    w_1 = 0
    w_2 = 0
    for ix in range(len(result)):
        w_0 += result[ix][0]
        w_1 += result[ix][1]
        w_2 += result[ix][2]
        
    return [w_0, w_1, w_2]

def gradientRiskcount(data, w):
    h = 1e-6
    w_grad = [0]*len(w)
    
    for i in range(len(w)):
        w_h = w[:]
        w_h[i] += h
        w_grad[i] = (riskCountWithScaled(data, w_h) - riskCountWithScaled(data, w))/h
    
    return w_grad

def getGradientDecent(func, f_gradient, data, _init_w):
    threshold = 1e-3
    _alpha = 0.25
    _beta = .75
    w = _init_w[:]
    _iter = 0
    
    while sum([a*b for a, b in zip(f_gradient(data, w), f_gradient(data, w))]) > threshold:
        t = 1
        _gradient = f_gradient(data, w)
        _delta_x = [a * -1 for a in _gradient]
        _dotProd = sum([a*b for a, b in zip(_gradient, _delta_x)])
        

        while not func(data, [b+c for b, c in zip(w, [a*t for a in _delta_x])]) < (func(data, w) + _alpha*t*_dotProd):
            t = t*_beta
            print 't: ', t
            
        w = [b+c for b,c in zip(w, [a*t for a in _delta_x])]
        plot_my_x_1.append(w[1])
        plot_my_x_2.append(w[2])
        _iter += 1

    return w

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

def addNoise(data, rou):
    arr = data[:]
    noise_index = random.sample(range(0, len(arr)-1), int(rou*len(arr)))
     
    for ix in range(len(noise_index)):
        if arr[noise_index[ix]][2] == -1:
            arr[noise_index[ix]][2] = 1
        else:
            arr[noise_index[ix]][2] = -1
    return arr

def myPlotContour(f):
    xlist = np.linspace(-10.0, 10.0, 1000)
    ylist = np.linspace(-10.0, 10.0, 1000)

    X, Y = np.meshgrid(xlist, ylist)
    Z = f([1,X,Y], [3,9,9])

    plt.figure()
    cp = plt.contour(X, Y, Z, 1)
    plt.clabel(cp, inline=True, 
               fontsize=10)
    plt.title('Contour Plot')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    
def myPlotScatter(data):
    positive_arr, negative_arr = getTwoDatasetForPlot(data)

    graph_positive_x1, graph_positive_x2, positive_label = subArrayColumn(positive_arr)
    graph_negative_x1, graph_negative_x2, negative_label = subArrayColumn(negative_arr)
    
    plt.scatter(graph_positive_x1,graph_positive_x2, marker='o', c='b', s=10, label='x_2')
    plt.scatter(graph_negative_x1,graph_negative_x2, marker='x', c='r', s=10, label='x_1')
    
    plt.title('Dataset')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    
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



      
if __name__ == "__main__":   
     
    clean_file_name = 'clean_' + str(SAMPLE_ITEM_NUM[1]) + '.txt' 
    clean_data = readTextFile(clean_file_name)
     
    noise_data = addNoise(clean_data, 0.1) 
    random.shuffle(noise_data)
    print noise_data
     
    OPTIMAL_W = getGradientDecent(riskCountWithScaled, getriskCountGradient, noise_data, INITIAL_W)
     
    print OPTIMAL_W
     
    myPlotScatter(train_noise_data)
    plt.plot(plot_my_x_1,plot_my_x_2)
     
    plt.show()
     
    print"=========================="
    total_risk_with_estimated_w = risk(test_noise_data, OPTIMAL_W)
    print total_risk_with_estimated_w


