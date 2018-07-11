'''
    Global variables
'''
ROU = 0  # mixed noise label ratio for generate data 
ROU_PLUS = 0.1
ROU_MINUS = 0.1
DATA_SAMPLE_RANGE = 1.0 # set the upper left and lower right corner
SAMPLE_ITEM_NUM = [100, 1000, 10000, 100000]   # set the total sample dots number
INITIAL_W = [0.9, 1, 1]   # set the initial w for gradient descent

# CLEAN_OUTPUT_FILE_NAME = 'clean_' + str(SAMPLE_ITEM_NUM) + '.txt' # output text file name
NOISE_OUTPUT_FILE_NAME = 'noise_' + str(SAMPLE_ITEM_NUM) + '_samples_and_rou_plus_' + str(ROU_PLUS) + '_samples_and_rou_minus' + str(ROU_MINUS) + '.txt' # output text file name

