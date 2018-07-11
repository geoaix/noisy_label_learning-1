from distutils.text_file import TextFile
from config import *
import matplotlib.pyplot as plt
import numpy as np
import random
from createDataFile import saveToFile
from old_main import myPlotScatter
from readFileToArray import readTextFile

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

if __name__ == "__main__":
    data = readTextFile(CLEAN_OUTPUT_FILE_NAME)
    
    noise_data = addNoise(data, ROU_NOISE)
#     print noise_data
#     myPlotScatter(noise_data)
#     plt.show()

    saveToFile(NOISE_OUTPUT_FILE_NAME, noise_data)
    