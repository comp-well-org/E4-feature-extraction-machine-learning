import pandas as pd
import scipy.signal as scisig
import os
import numpy as np

# Must be the same sample rate in EDA_Peak_Detection_Script.py
SAMPLE_RATE = 1

def get_user_input(prompt):
    try:
        return raw_input(prompt)
    except NameError:
        return input(prompt)


def getInputLoadFile(filepath):
    '''Asks user for type of file and file path. Loads corresponding data.

    OUTPUT:
        data:   DataFrame, index is a list of timestamps at 8Hz, columns include 
                AccelZ, AccelY, AccelX, Temp, EDA, filtered_eda
    '''
    filepath_confirm = os.path.join(filepath,"EDA.csv")
    data = loadData_E4(filepath)

    return data, filepath_confirm

def getOutputPath(fullOutputPath):
    if fullOutputPath[-4:] != '.csv':
        fullOutputPath = fullOutputPath+'.csv'
    return fullOutputPath

def _loadSingleFile_E4(filepath,list_of_columns, expected_sample_rate,freq):
    # Load data
    data = pd.read_csv(filepath)
    
    # Get the startTime and sample rate
    startTime = pd.to_datetime(float(data.columns.values[0]),unit="s")
    # startTime = float(data.columns.values[0])
    sampleRate = float(data.iloc[0][0])
    data = data[data.index!=0]
    data.index = data.index-1
    
    # Reset the data frame assuming expected_sample_rate
    data.columns = list_of_columns
    if sampleRate != expected_sample_rate:
        print('ERROR, NOT SAMPLED AT {0}HZ. PROBLEMS WILL OCCUR\n'.format(expected_sample_rate))

    # Make sure data has a sample rate of 8Hz
    data = interpolateDataToSampleHz(data,sampleRate,startTime)

    return data


def loadData_E4(filepath):
    # Load EDA data
    eda_data = _loadSingleFile_E4(os.path.join(filepath,'EDA.csv'),["EDA"],4,"250L")
    # Get the filtered data using a low-pass butterworth filter (cutoff:0.4hz, fs:sample_rate, order:6)
    eda_data['filtered_eda'] =  butter_lowpass_filter(eda_data['EDA'], 0.4, SAMPLE_RATE, 5)

    # Load ACC data
    acc_data = _loadSingleFile_E4(os.path.join(filepath,'ACC.csv'),["AccelX","AccelY","AccelZ"],32,"31250U")
    # Scale the accelometer to +-2g
    acc_data[["AccelX","AccelY","AccelZ"]] = acc_data[["AccelX","AccelY","AccelZ"]]
    acc_data["AccelMagnitude"] = np.sqrt(np.square(acc_data).sum(axis=1))

    # Load Temperature data
    temperature_data = _loadSingleFile_E4(os.path.join(filepath,'TEMP.csv'),["Temp"],4,"250L")

    data = eda_data.join(acc_data, how='outer')
    data = data.join(temperature_data, how='outer')

    # E4 sometimes records different length files - adjust as necessary
    min_length = min(len(acc_data), len(eda_data), len(temperature_data))

    return data[:min_length]


def interpolateDataToSampleHz(data,sample_rate,startTime):
    freq = str(1000//SAMPLE_RATE)+'L'
    if sample_rate<8:
        # Upsample by linear interpolation
        if sample_rate==2:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='500L')
        elif sample_rate==4:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='250L')
        data = data.resample(freq).mean()
    else:
        if sample_rate>8:
            # Downsample
            idx_range = list(range(0,len(data))) # TODO: double check this one
            data = data.iloc[idx_range[0::int(int(sample_rate)/8)]]
        # Set the index to be 4Hz
        data.index = pd.date_range(start=startTime, periods=len(data), freq=freq)

    # Interpolate all empty values
    data = interpolateEmptyValues(data)
    return data

def interpolateEmptyValues(data):
    cols = data.columns.values
    for c in cols:
        data.loc[:, c] = data[c].interpolate()

    return data

def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y