import numpy as np
import random
import math

# m denotes the number of examples here, not the number of features
# def gradientDescent(x, y, theta, alpha, m, numIterations):
#     xTrans = x.transpose()
#     for i in range(0, numIterations):
#         hypothesis = np.dot(x, theta)
#         loss = hypothesis - y
#         # avg cost per example (the 2 in 2*m doesn't really matter here.
#         # But to be consistent with the gradient, I include it)
#         cost = np.sum(loss ** 2) / (2 * m)
#         print("Iteration %d | Cost: %f" % (i, cost))
#         # avg gradient per example
#         gradient = np.dot(xTrans, loss) / m
#         # update
#         theta = theta - alpha * gradient
#     return theta

def gradientDescent(x, y, theta, alpha, m, numIterations):
    x = np.array(x)
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
#         cost = np.sum(loss ** 2) / (2 * m)
        cost = risk(x, theta, 0.4, 0.4)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
#         gradient = np.dot(xTrans, loss) / m
        gradient = getriskCountGradient(x, theta, 0.4, 0.4, 1)
        # update
#         theta = theta - alpha * gradient
        theta = [alpha*a for a in gradient]
    return theta

def getriskCountGradient(data, w, rou_plus, rou_minus, scaled_size):
    _n = len(data)
    result = []
    for _row in data:
        _y = _row[2]
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
        
    w_0 /= (_n*scaled_size)
    w_1 /= (_n*scaled_size)
    w_2 /= (_n*scaled_size)
#     print "update gradient to : ", [w_0, w_1, w_2]   
    return [w_0, w_1, w_2]

def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

def risk(data, w, rou_plus, rou_minus):
    _n = len(data)
    result = 0
    for _row in data:
        _y = _row[2]
        y_time_w_time_x = _y * sum([a*b for a,b in zip(w, _row)])
        _ratio_1 = (1 - rou_minus)
        _ratio_2 = (rou_plus*_y)
        row_1 = _ratio_1*math.log((1 + math.exp(-1*y_time_w_time_x)), math.e)
        row_2 = _ratio_2*math.log((1 + math.exp(y_time_w_time_x)), math.e)
        row_result = (row_1- row_2)/(math.log(2, math.e)*(1-rou_plus-rou_minus))
        result += row_result
    result = result/_n
    return result

def readTextFile(file_name):
    t_file = open(file_name)
    lines = t_file.readlines()
    result = []
#     print lines
    for line in lines:
#         result.append(np.asfarray(np.array(line.split()), float))
        _temp = []
        for _item in line.split():
            _temp.append(float(_item))
        result.append(_temp)
    return result

def dataTotwoArray(data):
    _x = []
    _y = []
    for _row in data:
        _a = _row[0]
        _b = _row[1]
        _c = _row[2]
        _x.append([_a,_b])
        _y.append(_c)
    return _x, _y

def addOneColumnToArray(data):
    _result = []
    for ix in range(len(data)):    
        _temp = []
        _temp.append(1)
        _temp.append(data[ix][0])
        _temp.append(data[ix][1])   
        _result.append(_temp)
        
    return _result

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
print x,y
m, n = np.shape(x)
print m, n
numIterations= 100
alpha = 0.0005
theta = [0.9, 1., 1.]
# theta = np.ones(n)

# print 'theta: ', theta
# theta = gradientDescent(x, y, theta, alpha, m, numIterations)
# print(theta)

test_noise_data = readTextFile('noise_output_data_1000_samples_and_rou_0.4.txt') 
x,y = dataTotwoArray(test_noise_data)

m, n = np.shape(x)
print x
x = addOneColumnToArray(x)
print x
print y
print m
print n
OPTIMAL_W = gradientDescent(x, y, theta, alpha, m, numIterations)
print OPTIMAL_W





