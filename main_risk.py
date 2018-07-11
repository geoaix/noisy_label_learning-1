from config import *
import math
import random
import matplotlib.pyplot as plt
from readFileToArray import readTextFile
from createNoiseFileFromClean import addNoise 


initial_w = INITIAL_W
scaled_size = 1

# Here rou_plus and rou_minus choose the same value for the noise should happen on both classes
rou_plus = 0.1
rou_minus = 0.1

plot_my_x_1 = []
plot_my_x_2 = []
plot_my_x_1.append(initial_w[1])
plot_my_x_2.append(initial_w[2])


def addNoiseDiffRou(data, rou_plus, rou_minus):
    arr = data[:]
    index_list_plus = []
    index_list_minus = []
    
    for ix in range(len(arr)):
        if arr[ix][2] == 1:
            index_list_plus.append(ix) 
        if arr[ix][2] == -1:
            index_list_minus.append(ix)
    
    sample_rou_num = len(arr)
    sample_rou_plus_num = int((sample_rou_num/2)*rou_plus)    
    noise_index_plus = random.sample(index_list_plus, sample_rou_plus_num)
    sample_rou_minus_num = int((sample_rou_num/2)*rou_minus)            
    noise_index_minus = random.sample(index_list_minus, sample_rou_minus_num)
    
    for index in noise_index_plus:
        arr[index][2] = -1
    for index in noise_index_minus:
        arr[index][2] = 1
    
    return arr

'''
    add x_0 to x and set it to 1 and w_0 is b
'''

def getCleanDataRisk(data, w):
    _n = len(data)
    result = 0
    for _row in data:
        _y = _row[2]
        
        _x = [1.]
        _x.append(_row[0])
        _x.append(_row[1])
        _row = _x
        
        y_time_w_time_x = _y * sum([a*b for a,b in zip(w, _row)])
        result += math.log((1 + math.exp(-1*y_time_w_time_x)), math.e)
        
    result = result/_n
        
    return result


def getLinearFunctionProd(x, w):
    _x = []
    _x.append(1)
    _x.append(x[0])
    _x.append(x[1])
#     print _x, w
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
    _update_count = 0
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
        _update_count += 1
    
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

    return w

def myPlotRiskWithDifRou(rou_arr, training_risk_arr, testing_risk_arr):
    plt.plot(rou_arr, training_risk_arr)
    plt.plot(rou_arr, testing_risk_arr)
    plt.legend(['training risk', 'testing risk'], loc='upper left')
    plt.title('Training Testing Risk Compare')
    plt.xlabel('rou_ratio')
    plt.ylabel('risk')

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

def getNoiseTrainingTestingRisk(_clean_data, _rou_plus, _rou_minus):
    _noise_data = addNoiseDiffRou(_clean_data, _rou_plus, _rou_minus) 
    random.shuffle(_noise_data)
    print 'shuffled noise data: ', _noise_data
    _training_data_size = int(len(_noise_data)*0.5)
    _my_training_data = _noise_data[:_training_data_size]
    _my_testing_data = _noise_data[_training_data_size:]
    print 'training data:', _my_training_data
    print len(_my_training_data)
    
    _OPTIMAL_W = getGradientDecent(riskCountWithScaled, getriskCountGradient, _my_training_data, initial_w)
    _w_name = 'f_hat_w_rouPlus_' + str(rou_plus) + '_rouMinus_' + str(rou_minus)
    print _w_name, _OPTIMAL_W
    
    _training_risk = risk(_my_training_data, _OPTIMAL_W)
    _testing_risk = risk(_my_testing_data, _OPTIMAL_W)
    
    return _training_risk, _testing_risk
    
if __name__ == "__main__": 
    rou_array = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    result_average_training_risk_array = []
    result_average_testing_risk_array = []
    
    '''
        In the final report the range is not 5 but 30
    '''
    for count in range(5): # Change to 30 for better result
        training_risk_array = []
        testing_risk_array = []
        
        clean_file_name = 'clean_' + str(SAMPLE_ITEM_NUM[1]) + '.txt' 
        clean_data = readTextFile(clean_file_name)
        print 'clean data size: ', len(clean_data)
        
        for item_rou in rou_array:
            rou_plus = item_rou
            rou_minus = item_rou       
            
            temp_train_risk, temp_test_risk = getNoiseTrainingTestingRisk(clean_data, rou_plus, rou_minus)
            training_risk_array.append(temp_train_risk)
            testing_risk_array.append(temp_test_risk)
        print training_risk_array
        result_average_training_risk_array.append(training_risk_array)
        print testing_risk_array
        result_average_testing_risk_array.append(testing_risk_array)
    
    training_risk_result = []
    for ix in range(len(result_average_training_risk_array[0])):
        training_risk_result.append(sum([a[ix] for a in result_average_training_risk_array])/len(result_average_training_risk_array))
    testing_risk_result = []
    for ix in range(len(result_average_testing_risk_array[0])):
        testing_risk_result.append(sum([a[ix] for a in result_average_testing_risk_array])/len(result_average_testing_risk_array))
    print training_risk_result
    print testing_risk_result 
    myPlotRiskWithDifRou(rou_array, training_risk_result, testing_risk_result)
    plt.show()
    
    
