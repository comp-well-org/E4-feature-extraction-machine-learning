import numpy as np
import pandas as pd
from hrvanalysis import remove_ectopic_beats, interpolate_nan_values, get_time_domain_features,\
    get_frequency_domain_features
from EDA_Peak_Detection_Script import getEDAfeatures


def extract_E4_features(fileroot, participant, settings_dict):
    """
    INPUT:
        fileroot: root of E4 data directory
        participant: number of each participants (i.e. participant 1,2,3,etc)
        settings_dict: a dictionary of parameters for EDA peak detection, including keys 'threshold','offset',
                        'rise time', and 'decay time'
    OUTPUT:
        A list of dataframe of EDA,ACC,TEMP features, dataframe of EDA peaks, dataframe of IBI features
    """
    # format eda features pd
    features_pd, peak_pd = getEDAfeatures(fileroot, '', settings_dict)
    features_pd.index.name = 'Time'
    features_pd = features_pd.reset_index(level=0)
    features_pd.insert(0, "participant", participant)
    # features_pd['Time'] = features_pd.Time.dt.tz_localize('UTC').dt.tz_convert('US/Central')

    # format peak pd
    peak_pd.index.name = 'Time'
    peak_pd = peak_pd.reset_index(level=0)
    peak_pd.insert(0, "participant", participant)

    # extract IBI
    data = readIBI(fileroot)
    # set the time window for IBI features here
    timewindow = 60
    ibi_features = getIBIfeatures(data,timewindow).rename(columns={'timestamp': 'Time'})
    ibi_features.insert(0, "participant", participant)
    ibi_features['Time'] = ibi_features['Time'].apply(lambda x: pd.to_datetime(x, unit='s'))

    return [features_pd,peak_pd,ibi_features]


def readIBI(fileroot):
    """
    INPUT:
        fileroot: root of E4 directory
    OUTPUT:
        A dataframe of IBI values mapped at timestamps
    """
    data = pd.read_csv(fileroot + '/IBI.csv', names=['timestamp', 'IBI'])
    if data.empty:
        print(fileroot + '/IBI.csv is empty')
        return data
    else:
        # Get the startTime and sample rate
        start_time = data.iloc[0][0]

    timestamps = [start_time]
    for i in range(len(data.values) - 2):
        timestamps.append(float(start_time + data.iloc[i + 1][0]))
    return pd.DataFrame(zip(timestamps, data.iloc[1:, 1]), columns=['timestamp', 'IBI'], index=None)


def getSingleIBIfeatures(data):
    """
    INPUT:
        data: Dataframe of IBI values mapped to timestamps
    OUTPUT:
        A single IBI feature vector
        For more information: https://aura-healthcare.github.io/hrvanalysis/hrvanalysis.html
    """
    if data.empty:
        return None
    IBI_data = data['IBI'].astype(float) * 1000
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=IBI_data, method="malik")
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    if not interpolated_nn_intervals[-1] > 1 and len(interpolated_nn_intervals) == 2:
        interpolated_nn_intervals[-1] = interpolated_nn_intervals[0]
    if not interpolated_nn_intervals[-1] > 1:
        interpolated_nn_intervals[-1] = np.median(interpolated_nn_intervals[1:-1])
    if not interpolated_nn_intervals[0] > 1:
        interpolated_nn_intervals[0] = np.median(interpolated_nn_intervals[1:-1])
    # get features
    time_features = get_time_domain_features(interpolated_nn_intervals)
    freq_features = get_frequency_domain_features(interpolated_nn_intervals, method='lomb')
    IBI_features_df = pd.DataFrame({**time_features, **freq_features}, index=[0])
    # IBI_features_df.insert(0, "participant", participant)
    return IBI_features_df


def getIBIfeatures(data,time_window):
    """
        INPUT:
            data: Dataframe of IBI values mapped to timestamps
        OUTPUT:
            IBI features
            For more information: https://aura-healthcare.github.io/hrvanalysis/hrvanalysis.html
    """
    timestamp = data.timestamp.values
    IBI_data = np.array(data['IBI'].astype(float) * 1000)

    time_features_nn = np.zeros((1, 16))
    freq_features_nn = np.zeros((1, 7))
    timestamps = [0]
    for t in timestamp:
        if t >= timestamp[-1] - time_window:
            break
        curr_time = round(t + time_window)
        if curr_time in timestamps:
            continue
        timestamps.append(pd.to_datetime(curr_time, unit='s'))
        index_less = timestamp <= (t + time_window)
        index_larger = timestamp >= t
        index = index_less & index_larger
        curr_rr_interval = IBI_data[index]

        # This remove ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(rr_intervals=curr_rr_interval, method="malik")
        # This replace ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
        if not interpolated_nn_intervals[-1] > 1 and len(interpolated_nn_intervals) == 2:
            interpolated_nn_intervals[-1] = interpolated_nn_intervals[0]
        if not interpolated_nn_intervals[-1] > 1:
            interpolated_nn_intervals[-1] = np.median(interpolated_nn_intervals[1:-1])
        if not interpolated_nn_intervals[0] > 1:
            interpolated_nn_intervals[0] = np.median(interpolated_nn_intervals[1:-1])

        time_domain_features = get_time_domain_features(interpolated_nn_intervals)
        time_features_nn = np.vstack((time_features_nn, np.array([time_domain_features['mean_nni'],
                                                                  time_domain_features['sdnn'],
                                                                  time_domain_features['sdsd'],
                                                                  time_domain_features['nni_50'],
                                                                  time_domain_features['pnni_50'],
                                                                  time_domain_features['nni_20'],
                                                                  time_domain_features['pnni_20'],
                                                                  time_domain_features['rmssd'],
                                                                  time_domain_features['median_nni'],
                                                                  time_domain_features['range_nni'],
                                                                  time_domain_features['cvsd'],
                                                                  time_domain_features['cvnni'],
                                                                  time_domain_features['mean_hr'],
                                                                  time_domain_features['max_hr'],
                                                                  time_domain_features['min_hr'],
                                                                  time_domain_features['std_hr']]).reshape(1, 16)))
        freq_domain_features = get_frequency_domain_features(interpolated_nn_intervals,
                                                             method='lomb')
        freq_features_nn = np.vstack((freq_features_nn, np.array([freq_domain_features['lf'],
                                                                  freq_domain_features['hf'],
                                                                  freq_domain_features['lf_hf_ratio'],
                                                                  freq_domain_features['lfnu'],
                                                                  freq_domain_features['hfnu'],
                                                                  freq_domain_features['total_power'],
                                                                  freq_domain_features['vlf']]).reshape(1, 7)))
    IBI_features = np.hstack((np.array(timestamps[1:]).reshape((-1, 1)),
                              time_features_nn[1:, :],
                              freq_features_nn[1:, :]))
    IBI_features_df = pd.DataFrame(IBI_features, columns=['timestamp', 'mean_nni', 'sdnn', 'sdsd', 'nni_50',
                                                          'pnni_50', 'nni_20', 'pnni_20', 'rmssd', 'median_nni',
                                                          'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr',
                                                          'std_hr', 'lf', 'hf', 'lf_hf_ratio', 'lfnu',
                                                          'hfnu', 'total_power', 'vlf'])
    # IBI_features_df.insert(0, "participant", participant)
    return IBI_features_df

