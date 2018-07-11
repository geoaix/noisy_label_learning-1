import math
from config import *
from readFileToArray import readTextFile

def f(x, w):
    _x = []
    _x.append(1)
    _x.append(x[0])
    _x.append(x[1])

    return sum([v_1*v_2 for v_1, v_2 in zip(_x, w)])

def getLogisticLoss(data, w):
    _n = len(data)
    result = 0
    for _row in data:
        _y = _row[2]
        _x = [_row[0], _row[1]]
        _inside = 1 + math.exp(-1*_y*f(_x, w))
        result += math.log(_inside, math.e)/math.log(2, math.e)
    result = result/_n
    return result

def getOnelogisticLoss(data, w):
    _inside = 1 + math.exp(-1*oneLabel*f(oneData, w))
    return math.log(_inside, math.e)/math.log(2, math.e)

def risk(data, w, rou):
    _n = len(data)
    result = 0
    for _row in data:
        _y = _row[2]
        y_time_w_time_x = _y * sum([a*b for a,b in zip(w, _row)])
        _ratio_1 = (1 - rou)
        _ratio_2 = (rou*_y)
        row_1 = _ratio_1*math.log((1 + math.exp(-1*y_time_w_time_x)), math.e)
        row_2 = _ratio_2*math.log((1 + math.exp(y_time_w_time_x)), math.e)
        row_result = (row_1- row_2)/(math.log(2, math.e)*(1-rou-rou))
        result += row_result
    result = result/_n
    return result


if __name__ == "__main__": 
    w_rou_0 = [0.011713513702629217, -0.0019092349782635649, -0.02275299829006938]
    w_rou_0_1 = [-0.05314632440761786, 0.9399758904892336, 3.34726300063994]
    w_rou_0_2 = [-0.10228626318452379, 0.5946416445807761, 3.8474458317553184]
    w_rou_0_3 = [0.7060770666965366, 0.9683793140938844, 2.2213578620730843]
    w_rou_0_4 = [0.26249154210577463, 0.45496587081407436, 3.915487039570065]
    print "---------------------Estimate 1000 Samples-------------------------------"
    test_clean_data = readTextFile('clean_output_data_1000_samples_and_rou_0.txt') 
#     print test_noise_data
    clean_total_risk = risk(test_clean_data, w_rou_0, 0)
    print 'risk with 0 rou: ', clean_total_risk

    
    test_noise_data = readTextFile('noise_output_data_1000_samples_and_rou_0.1.txt') 
#     print test_noise_data
    noise_total_risk = risk(test_noise_data, w_rou_0_1, 0.1)
    print 'risk with 0.1 rou: ', noise_total_risk
    print 'difference with clean: ', (noise_total_risk-clean_total_risk)/clean_total_risk
    
    test_noise_data = readTextFile('noise_output_data_1000_samples_and_rou_0.2.txt') 
#     print test_noise_data
    noise_total_risk = risk(test_noise_data, w_rou_0_2, 0.2)
    print 'risk with 0.2 rou: ', noise_total_risk
    print 'difference with clean: ', (noise_total_risk-clean_total_risk)/clean_total_risk
    
    test_noise_data = readTextFile('noise_output_data_1000_samples_and_rou_0.3.txt') 
#     print test_noise_data
    noise_total_risk = risk(test_noise_data, w_rou_0_3, 0.3)
    print 'risk with 0.3 rou: ', noise_total_risk
    print 'difference with clean: ', (noise_total_risk-clean_total_risk)/clean_total_risk
    
    test_noise_data = readTextFile('noise_output_data_1000_samples_and_rou_0.4.txt') 
#     print test_noise_data
    noise_total_risk = risk(test_noise_data, w_rou_0_4, 0.4)
    print 'risk with 0.4 rou: ', noise_total_risk
    print 'difference with clean: ', (noise_total_risk-clean_total_risk)/clean_total_risk
    
    