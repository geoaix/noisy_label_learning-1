from readFileToArray import readTextFile
from createNoiseFileFromClean import addNoiseDiffRou
from config import *
from main import myPlotScatter
import matplotlib.pyplot as plt

def checkData(data):
    arr = data[:]
    index_list_plus = []
    index_list_minus = []
    
    for ix in range(len(arr)):
        if arr[ix][2] == 1:
            index_list_plus.append(ix) 
        if arr[ix][2] == -1:
            index_list_minus.append(ix)
    
    print 'index_plus', index_list_plus 
    print len(index_list_plus)
    print 'index_minus', index_list_minus
    print len(index_list_minus)
    print abs((len(index_list_plus)- len(index_list_minus)))/float(len(index_list_minus))

clean_file_name = 'clean_' + str(SAMPLE_ITEM_NUM[1]) + '.txt' 
clean_data = readTextFile(clean_file_name)

checkData(clean_data)
print "=================================================="
noise_data = addNoiseDiffRou(clean_data, 0.1, 0.1) 
checkData(noise_data)

myPlotScatter(noise_data)
plt.show()
