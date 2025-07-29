import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrograms(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_sources = ['data1', 'data2', 'data3', 'data4']
    for source in data_sources:
        source_input_folder = os.path.join(input_folder, source)
        source_output_folder = os.path.join(output_folder, source)
        if not os.path.exists(source_output_folder):
            os.makedirs(source_output_folder)
        
        for wav_file in os.listdir(source_input_folder):
            if wav_file.endswith('.wav'):
                wav_file_path = os.path.join(source_input_folder, wav_file)
                y, sr = librosa.load(wav_file_path, sr=None)  # Load the audio file with its original sample rate
                
                # Compute the spectrogram
                S = librosa.stft(y)  # Short-Time Fourier Transform
                S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)  # Convert amplitude to dB
                
                # Plot the spectrogram
                plt.figure(figsize=(10, 6))
                librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
                #plt.axis('off')  # Turn off the axes
                #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove white padding
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Spectrogram')
                # Add a color bar to indicate the amplitude in dB
                cbar = plt.colorbar(format='%+2.0f dB')
                cbar.set_label('Amplitude (dB)')

                # Define the path to save the spectrogram image
                spectrogram_image_path = os.path.join(source_output_folder, wav_file.replace('.wav', '.png'))
                
                # Save the spectrogram as an image file
                plt.savefig(spectrogram_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                #print(f'Spectrogram image saved at {spectrogram_image_path}')

# Example usage
input_folder = "0_long_data/Juni15/Train_wav_data_21_May_short_full_wav"#'Train_wav_data_21_May_1second_noUpSampling'  # Replace with your input folder path
output_folder = "0_long_data/Juni15/Train_wav_data_21_May_short_full_wav_spectrograms" #"0_long_data/Juni15/Train_wav_data_21_May_lowpass_spectrogram_withAxis_hopefully_edgescut_spectrogram" 

create_spectrograms(input_folder, output_folder)