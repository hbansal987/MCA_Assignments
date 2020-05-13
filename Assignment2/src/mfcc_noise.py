# -*- coding: utf-8 -*-
"""MFCC_Noise.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bD70l8wMS-EJH5uszuItfaJvoHyznTPy
"""

from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct
import math
import glob
import pickle
import time
from pathlib import Path
from random import seed
from random import randint

#folder_list = ['/content/drive/My Drive/Dataset/training/eight','/content/drive/My Drive/Dataset/training/five',
#               '/content/drive/My Drive/Dataset/training/four','/content/drive/My Drive/Dataset/training/nine',
#               '/content/drive/My Drive/Dataset/training/one','/content/drive/My Drive/Dataset/training/seven',
#               '/content/drive/My Drive/Dataset/training/six','/content/drive/My Drive/Dataset/training/three',
#               '/content/drive/My Drive/Dataset/training/two','/content/drive/My Drive/Dataset/training/zero']
folder_list = ['/content/drive/My Drive/Dataset/training/eight']

sf1,noise1 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/doing_the_dishes.wav')
sf2,noise2 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/dude_miaowing.wav')
sf3,noise3 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/exercise_bike.wav')
sf4,noise4 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/pink_noise.wav')
sf5,noise5 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/running_tap.wav')
sf6,noise6 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/white_noise.wav')
noise_list = [noise1,noise2,noise3,noise4,noise5,noise6]

X_train = []
y_train = []
for folder in folder_list:
  count= 0
  path = folder + "/*.wav"
  for ts in glob.glob(path):
    #if(count==1):
    #  break
    """Getting the input"""
    sampling_frequency, audio = scipy.io.wavfile.read(ts)
    value = randint(0,5)
    print("value ",value)
    noise_chosen = noise_list[value]
    index = randint(0,800000)
    result = 0.5 * audio[:len(audio)] + 0.5 * noise_chosen[index:index+len(audio)]
    audio = result

    fourier_size = 1024
    stride_size = 5
    frame_audio = divide_audio_frames(audio,fourier_size,stride_size,sampling_frequency)
    #plt.plot(frame_audio[1,:])
    #plt.show()

    """Converting the starting and endings to close to 0 (so that it appers that each window is a seperate audio)"""
    frame_audio *= np.hamming(fourier_size)
    #plt.plot(frame_audio[1,:])
    #plt.show()
    #print(frame_audio.shape)

    """Applying the fourier transform on the windows"""
    NFFT = 1024 #Fourier Size
    frames_magnitude = np.absolute(np.fft.rfft(frame_audio,NFFT))
    frames_power = (1/NFFT) * np.square(frames_magnitude)
    #print(frames_power.shape)
    #for i in range (50):
    #  plt.plot(frames_power[10,:])  
    #  plt.show() 

    """Computing the mel filter bank"""
    """This will give us the power of each frequency band"""
    min_frequency = 0
    max_frequency = sampling_frequency/2
    number_mel_filter = 13

    """getting the filter points and frequencies in mel form"""
    filter_points , mel_freqs = get_filter_points(min_frequency,max_frequency,number_mel_filter,fourier_size,sampling_frequency)
    #print(filter_points)
    #print(mel_freqs)

    """Making of the filters"""
    filter_array = construct_filters(filter_points,fourier_size)
    
    #enorm = 2.0 / (mel_freqs[2:number_mel_filter+2] - mel_freqs[:number_mel_filter])
    #filter_array *= enorm[:, np.newaxis]

    filtered_audio = np.dot(frames_power,filter_array.T)
    audio_log = 10 * np.log10(filtered_audio)
    #audio_log = filtered_audio
    #print("audio_log.shape ",audio_log.shape)
    #plt.plot(audio_log)
    #plt.show()

    num_ceps = 12
    mfcc = dct(audio_log, type=2, axis=1, norm= 'ortho')[:,1 : (num_ceps + 1)] # Keep 2-13
    #print("mfcc.shape ",mfcc.shape)
    #plt.plot(mfcc)
    #plt.show()
    X_train.append(mfcc.ravel())
    folder_array = folder.split("/")
    y_train.append(folder_array[6])
    count=count+1
    print("count ",count)

#output = open('/content/drive/My Drive/Pickle_MFCC_X_train2'+ '.pickle', 'wb')
#pickle.dump(X_train, output)
#output2 = open('/content/drive/My Drive/Pickle_MFCC_y_train2'+ '.pickle', 'wb')
#pickle.dump(y_train, output2)

print(X_train[0])
print(X_train[0].shape)

from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct
import math
import glob
import pickle
import time
from pathlib import Path

#folder_list = ['/content/drive/My Drive/Dataset/training/eight','/content/drive/My Drive/Dataset/training/five',
#               '/content/drive/My Drive/Dataset/training/four','/content/drive/My Drive/Dataset/training/nine',
#               '/content/drive/My Drive/Dataset/training/one','/content/drive/My Drive/Dataset/training/seven',
#               '/content/drive/My Drive/Dataset/training/six','/content/drive/My Drive/Dataset/training/three',
#               '/content/drive/My Drive/Dataset/training/two','/content/drive/My Drive/Dataset/training/zero']

sf1,noise1 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/doing_the_dishes.wav')
sf2,noise2 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/dude_miaowing.wav')
sf3,noise3 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/exercise_bike.wav')
sf4,noise4 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/pink_noise.wav')
sf5,noise5 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/running_tap.wav')
sf6,noise6 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/white_noise.wav')

print(len(noise1))
print(len(noise2))
print(len(noise3))
print(len(noise4))
print(len(noise5))
print(len(noise6))

from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct
import math
import glob
import pickle
import time
from pathlib import Path
infile = open('/content/drive/My Drive/Pickle_MFCC_X_train2.pickle','rb')
X_train = pickle.load(infile)
print(len(X_train))

infile2 = open('/content/drive/My Drive/Pickle_MFCC_y_train2.pickle','rb')
y_train = pickle.load(infile2)
print(len(y_train))

"""Divinding in small frames"""
def divide_audio_frames(audio , fourier_size, stride_size, sample_rate):
  
  pad_size = int(fourier_size/2)
  audio = np.pad(audio,pad_size,mode = 'reflect')
  
  frame_length = sample_rate*stride_size/1000
  frame_length = int(np.round(frame_length))
  
  frame_number = int(((len(audio) - fourier_size)/frame_length) + 1)
  frame_array = np.zeros((frame_number,fourier_size))

  for i in range (frame_number):
    frame_array[i] = audio[i*frame_length:i*frame_length + fourier_size]

  return(frame_array)

def mel_to_frequency(mels):
  return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def frequency_to_mel(freq):
  return 2595.0 * np.log10(1.0 + freq / 700.0)

def get_filter_points(min_frequency,max_frequency,number_mel_filter,fourier_size,sampling_frequency):
  min_mel = frequency_to_mel(min_frequency)
  max_mel = frequency_to_mel(max_frequency)

  """Evenly spaces numbers (number_mel_filter+2) in the interval min_mel and max_mel"""
  mel_array = np.linspace(min_mel,max_mel,num=number_mel_filter+2)

  freq = mel_to_frequency(mel_array)
  me = np.floor((fourier_size+1) / sampling_frequency*freq)

  return me,freq

"""Constructing the filter bank"""
def construct_filters(filter_points,fourier_size):
  filter_array = np.zeros((len(filter_points)-2,int(fourier_size/2 +1)))

  for n in range (len(filter_points)-2):
    change1 = int(filter_points[n + 1]) - int(filter_points[n])
    filter_array[n, int(filter_points[n]) : int(filter_points[n + 1])] = np.linspace(0, 1, change1)
    change2 = int(filter_points[n + 2]) - int(filter_points[n + 1])
    filter_array[n, int(filter_points[n + 1]) : int(filter_points[n + 2])] = np.linspace(1, 0, change2)
  
  return filter_array

from sklearn import svm

#Create a svm Classifier
for i in range(len(X_train)):
  #print(X_train[i].shape)
  if(X_train[i].shape[0] < 2412):
    temp = np.zeros(2412)
    for k in range(X_train[i].shape[0]):
      temp[k] = (X_train[i])[k]
    for k in range(2412 - X_train[i].shape[0]):
      temp[X_train[i].shape[0] + k] = 0
    
    X_train[i] = temp

    print("HELLLOOOO ",X_train[i].shape)
  X_train[i] = np.nan_to_num(X_train[i])
  X_train[i] = np.array(X_train[i], dtype=np.float64)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

folder_list = ['/content/drive/My Drive/Dataset/validation/eight','/content/drive/My Drive/Dataset/validation/five',
               '/content/drive/My Drive/Dataset/validation/four','/content/drive/My Drive/Dataset/validation/nine',
               '/content/drive/My Drive/Dataset/validation/one','/content/drive/My Drive/Dataset/validation/seven',
               '/content/drive/My Drive/Dataset/validation/six','/content/drive/My Drive/Dataset/validation/three',
               '/content/drive/My Drive/Dataset/validation/two','/content/drive/My Drive/Dataset/validation/zero']
X_test = []
y_test = []
for folder in folder_list:
  count= 0
  path = folder + "/*.wav"
  for ts in glob.glob(path):
    #if(count==1):
    #  break
    #ts = '/content/drive/My Drive/Dataset/training/eight/004ae714_nohash_0.wav'
    #Audio(ts)
    """Getting the input"""
    sampling_frequency, audio = scipy.io.wavfile.read(ts)
    value = randint(0,5)
    print("value ",value)
    noise_chosen = noise_list[value]
    index = randint(0,800000)
    result = 0.5 * audio[:len(audio)] + 0.5 * noise_chosen[index:index+len(audio)]
    audio = result

    fourier_size = 1024
    stride_size = 5
    frame_audio = divide_audio_frames(audio,fourier_size,stride_size,sampling_frequency)
    #plt.plot(frame_audio[1,:])
    #plt.show()

    """Converting the starting and endings to close to 0"""
    frame_audio *= np.hamming(fourier_size)
    #plt.plot(frame_audio[1,:])
    #plt.show()
    #print(frame_audio.shape)

    NFFT = 1024
    frames_magnitude = np.absolute(np.fft.rfft(frame_audio,NFFT))
    frames_power = (1/NFFT) * np.square(frames_magnitude)
    print(frames_power.shape)
    #for i in range (50):
    #plt.plot(frames_power[10,:])  
    #plt.show() 

    min_frequency = 0
    max_frequency = sampling_frequency/2
    number_mel_filter = 13

    filter_points , mel_freqs = get_filter_points(min_frequency,max_frequency,number_mel_filter,fourier_size,sampling_frequency)
    #print(filter_points)
    #print(mel_freqs)

    filter_array = construct_filters(filter_points,fourier_size)
    enorm = 2.0 / (mel_freqs[2:number_mel_filter+2] - mel_freqs[:number_mel_filter])
    filter_array *= enorm[:, np.newaxis]
    plt.figure(figsize=(15,4))
    #for n in range(filter_array.shape[0]):
    #    plt.plot(filter_array[n])

    filtered_audio = np.dot(frames_power,filter_array.T)
    audio_log = 10 * np.log10(filtered_audio)
    #audio_log = filtered_audio
    #print("audio_log.shape ",audio_log.shape)
    #plt.plot(audio_log)
    #plt.show()

    num_ceps = 12
    mfcc = dct(audio_log, type=2, axis=1, norm= 'ortho')[:,1 : (num_ceps + 1)] # Keep 2-13
    #print("mfcc.shape ",mfcc.shape)
    #plt.plot(mfcc)
    #plt.show()
    X_test.append(mfcc.ravel())
    folder_array = folder.split("/")
    y_test.append(folder_array[6])
    count=count+1
    print("count ",count)

from sklearn import metrics
for i in range(len(X_test)):
  #print(X_train[i].shape)
  if(X_test[i].shape[0] < 2412):
    temp = np.zeros(2412)
    for k in range(X_test[i].shape[0]):
      temp[k] = (X_test[i])[k]
    for k in range(2412 - X_test[i].shape[0]):
      temp[X_test[i].shape[0] + k] = 0
    
    X_test[i] = temp

    print("HELLLOOOO ",X_test[i].shape)
  X_test[i] = np.nan_to_num(X_test[i])
  X_test[i] = np.array(X_test[i], dtype=np.float64)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred, average = 'macro'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'macro'))

#from scikits.audiolab import wavread, wavwrite

"""sf, audio = scipy.io.wavfile.read("/content/drive/My Drive/Dataset/training/eight/004ae714_nohash_0.wav")
sf1,noise1 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/doing_the_dishes.wav')

#assert fs1 == fs2
#assert enc1 == enc2
result = 0.5 * audio[:len(audio)] + 0.5 * noise1[300:300+len(audio)]

scipy.io.wavfile.write('/content/drive/My Drive/Dataset/Noise_sounds/result.wav',sf,result)"""

"""from IPython.display import Audio
sf3,n =  scipy.io.wavfile.read('/content/drive/My Drive/Dataset/Noise_sounds/result.wav')
Audio(result,rate=sf3)"""

