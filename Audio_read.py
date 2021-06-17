import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import csv
import pathlib



# To read and store audio files & spectogram 
A=[]
B=[]
cmap = plt.get_cmap('viridis')
plt.figure(figsize=(8,8))

# To create csv file 
header = 'filename spectogram label'
header = header.split()
file = open('dataset_2.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)


#Reading all files
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'test_data_1/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'/Users/jaya/Documents/College/Final_project/genres/{g}'):
        songname = f'/Users/jaya/Documents/College/Final_project/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True)
        A.append((songname))
        
        # Spectogram of all files
        ps = librosa.feature.melspectrogram(y= y,n_mels=128, sr=44100, win_length=2048,hop_length=512 )
        B.append((ps,songname))
        
        #Save specto of all files
        plt.specgram(ps);
        plt.axis('off');
        plt.savefig(f'test_data_1/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()

        # Add data to csv file
        to_append = f'{filename}  {np.mean(ps)} {g}'    
        file = open('dataset_2.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())