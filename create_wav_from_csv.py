import pandas as pd
import numpy as np
import os
from scipy.io.wavfile import write

def create_wav_file_from_csv(csv_path, output_folder, sample_rate=44100):
    df = pd.read_csv(csv_path)
    
    # create the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data_sources = ['data1', 'data2', 'data3', 'data4']
    
    for source in data_sources:
        source_data = df[source].values
        
        # calculate total number of samples based on the time value in the last row
        total_time = df['time'].iloc[-1]
        total_samples = int(total_time * sample_rate)
        
        # resample data to match the sample rate and the total length of the data
        resampled_data = np.interp(
            np.linspace(0, len(source_data), total_samples, endpoint=False),
            np.arange(len(source_data)),
            source_data
        )
        
        # convert to int16 format
        int_data = np.int16(resampled_data / np.max(np.abs(resampled_data)) * 32767)
        
        wav_filename = f"{source}.wav"
        wav_filepath = os.path.join(output_folder, wav_filename)
        write(wav_filepath, sample_rate, int_data)

csv_path = '0_long_data/Juni15/0_FINALTrain_data_21_May_long_DataGathering__100000_delay1_synchronized_short.csv'
output_folder = '0_long_data/Juni15/Train_wav_data_21_May_short_full_wav'

create_wav_file_from_csv(csv_path, output_folder)
