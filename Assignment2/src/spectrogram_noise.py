# -*- coding: utf-8 -*-
"""Spectrogram_Noise.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1efbmlCbx3CXSZskIlVEdrCRj1BUQhvYA
"""

!unzip -uq "/content/drive/My Drive/assignment2.zip"

from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import math
import glob
import pickle
import time
from pathlib import Path
from random import seed
from random import randint

folder_list = ['/content/drive/My Drive/Dataset/training/eight','/content/drive/My Drive/Dataset/training/five',
               '/content/drive/My Drive/Dataset/training/four','/content/drive/My Drive/Dataset/training/nine',
               '/content/drive/My Drive/Dataset/training/one','/content/drive/My Drive/Dataset/training/seven',
               '/content/drive/My Drive/Dataset/training/six','/content/drive/My Drive/Dataset/training/three',
              '/content/drive/My Drive/Dataset/training/two','/content/drive/My Drive/Dataset/training/zero']
#folder_list = ['/content/drive/My Drive/Dataset/training/eight']

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
    try:
      #if(count==1):
      #  break
      """Getting the input"""
      sampling_frequency, audio = scipy.io.wavfile.read(ts)
      value = randint(0,5)
      print("value ",value)
      noise_chosen = noise_list[value]
      index = randint(0,800000)
      result = 0.5*audio[:len(audio)] + 0.5*noise_chosen[index:index+len(audio)]
      audio = result
      #Audio(audio,rate = sampling_frequency)

      """Converting the input size in power of 2 (Needed for FFT)"""
      #next_length = next_power_of_2(length)
      #audio_main = []
      #for i in range (length):
      #  audio_main.append(audio[i])
      #for i in range(next_length - length):
      #  audio_main.append(0)

      audio_main = audio
      #audio_fft = fourier_transform(audio_main)

      starts_array, specto = spectrogram_make(audio_main,sampling_frequency)
      specto2 = specto.ravel()

      X_train.append(specto2)
      folder_array = folder.split("/")
      y_train.append(folder_array[6])
      print("count ",count)
      count=count+1
    except:
      print("Some error")

print(X_train[0])

import pickle
output = open('/content/drive/My Drive/Pickle_X_train2_noise'+ '.pickle', 'wb')
pickle.dump(X_train, output)
output2 = open('/content/drive/My Drive/Pickle_y_train2_noise'+ '.pickle', 'wb')
pickle.dump(y_train, output2)

import pickle
infile = open('/content/drive/My Drive/Pickle_X_train2_noise.pickle','rb')
X_train = pickle.load(infile)
print(len(X_train))
infile2 = open('/content/drive/My Drive/Pickle_y_train2_noise.pickle','rb')
y_train = pickle.load(infile2)
print(len(y_train))

from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import math
import glob
import pickle
import time
from pathlib import Path
from random import seed
from random import randint

"""Fast Fourier Transform (Not using in this question as when I convert to power of 0 it gives problems)"""
"""def fourier_transform(audio):
  length = len(audio)
  audio = np.asarray(audio)
  if(length < 2):
    length_min = length
  else:
    length_min = 2
  
  n = np.arange(length_min)
  #print(n)
  k = n[:, None]
  #print(k)
  temp = -2j * np.pi * n * k / length_min
  M = np.exp(temp)
  #print(M)
  X = np.dot(M, audio.reshape((length_min, -1)))

  while X.shape[0] < length:
    X_odd = X[:, int(X.shape[1] / 2):]
    X_even = X[:, :int(X.shape[1] / 2)]
    terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
    X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])
    
  return X.ravel()"""

"""Converting length of audio as power of 2"""
def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

"""Change"""
def spectrogram_make(audio,sampling_frequency):
  stride = 10
  window = 20
  max_frequency = sampling_frequency

  stride_size = int((1/1000)*stride*sampling_frequency)
  window_size = int((1/1000)*sampling_frequency*window)
  
  stride_size = 128
  window_size = 64
  """Forming the array with gives the start points for different windows (kept at a size of 64 seperation)"""
  starts = np.arange(0,len(audio),window_size)
  fourier = []
  for start in starts:
    """Making the window with 128 seperation (So that there is overlap)"""
    if(start < len(audio) - 128):
      window = get_fourier(audio[start:start+stride_size])
      fourier.append(window)
  
  output = np.transpose(np.array(fourier))
  output = 20*np.log10(output)
  """If dosent work then make it 64 64"""
  return(starts,output)

"""Using DFT as in FFT we need the size of the array to be the power of 2 which was causeing problem in dimensions"""
def get_fourier(audio):
  fourier =[]
  length = len(audio)
  NyLimit = int(length/2)

  for i in range (NyLimit):
    
    arr = np.zeros(length)
    for w in range(length):
      arr[w] = w
    #arr = np.arange(0,length,1)
    value = 1j*2*np.pi*arr*i
    fourier_value = np.sum(audio*np.exp(value/length))
    fourier_value = fourier_value/length
    fourier_value = np.abs(fourier_value)
    fourier_value = fourier_value*2
    fourier.append(fourier_value)
  
  return fourier

infile = open('/content/drive/My Drive/Pickle_X_train2.pickle','rb')
best_model2 = pickle.load(infile)
print(len(best_model2))

"""For plotting the spectrogram (Taken directly. Just for checking purposes)"""
"""def get_Hz_scale_vec(ks,sample_rate,Npoints):
    freq_Hz = ks*sample_rate/Npoints
    freq_Hz  = [int(i) for i in freq_Hz ] 
    return(freq_Hz )

def plot_spectrogram(spec,sample_rate, L, starts, mappable = None):
    plt.figure(figsize=(20,8))
    plt_spec = plt.imshow(spec,origin='lower')

    ## create ylim
    Nyticks = 10
    ks      = np.linspace(0,len(spec),Nyticks)
    ksHz    = get_Hz_scale_vec(ks,sample_rate,len(audio_main))
    plt.yticks(ks,ksHz)
    plt.ylabel("Frequency (Hz)")

    ## create xlim
    Nxticks = 10
    ts_spec = np.linspace(0,len(spec),Nxticks)
    ts_spec_sec  = ["{:4.2f}".format(i) for i in np.linspace(0,1*starts[-1]/len(spec),Nxticks)]
    plt.xticks(ts_spec,ts_spec_sec)
    plt.xlabel("Time (sec)")

    plt.title("Spectrogram L={} Spectrogram.shape={}".format(L,len(spec)))
    plt.colorbar(mappable,use_gridspec=True)
    plt.show()
    return(plt_spec)
plot_spectrogram(specto,sampling_frequency,128, starts_array)"""

print(X_train[2].shape)

from sklearn import svm

#Create a svm Classifier
for i in range(len(X_train)):
  #print(X_train[i].shape)
  if(X_train[i].shape[0] < 15872):
    temp = np.zeros(15872)
    for k in range(X_train[i].shape[0]):
      temp[k] = (X_train[i])[k]
    for k in range(15872 - X_train[i].shape[0]):
      temp[X_train[i].shape[0] + k] = 0
    
    X_train[i] = temp

    print("HELLLOOOO ",X_train[i].shape)
  #print("old ",X_train[i])
  for y in range (8000):
    if((X_train[i])[y] == float('inf') or (X_train[i])[y] == -float('inf')):
      #print("Hi")
      (X_train[i])[y] = 0
  #print("new ",X_train[i])
  X_train[i] = np.nan_to_num(X_train[i])
  #X_train[i] = np.array(X_train[i], dtype=np.float64)
clf = svm.SVC(kernel='poly')

clf.fit(X_train, y_train)

import pickle
output3 = open('/content/drive/My Drive/Pickle_Model_ques1_2'+ '.pickle', 'wb')
pickle.dump(clf, output3)

folder_list = ['/content/drive/My Drive/Dataset/validation/eight','/content/drive/My Drive/Dataset/validation/five',
               '/content/drive/My Drive/Dataset/validation/four','/content/drive/My Drive/Dataset/validation/nine',
               '/content/drive/My Drive/Dataset/validation/one','/content/drive/My Drive/Dataset/validation/seven',
               '/content/drive/My Drive/Dataset/validation/six','/content/drive/My Drive/Dataset/validation/three',
               '/content/drive/My Drive/Dataset/validation/two','/content/drive/My Drive/Dataset/validation/zero']
sf1,noise1 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/doing_the_dishes.wav')
sf2,noise2 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/dude_miaowing.wav')
sf3,noise3 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/exercise_bike.wav')
sf4,noise4 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/pink_noise.wav')
sf5,noise5 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/running_tap.wav')
sf6,noise6 = scipy.io.wavfile.read('/content/drive/My Drive/Dataset/_background_noise_/white_noise.wav')
noise_list = [noise1,noise2,noise3,noise4,noise5,noise6]

X_test = []
y_test = []
for folder in folder_list:
  count= 0
  path = folder + "/*.wav"
  for ts in glob.glob(path):
    try:
      #if(count==10):
      #  break
      #ts = '/content/drive/My Drive/Dataset/training/eight/004ae714_nohash_0.wav'
      #Audio(ts)
      """Getting the input"""
      sampling_frequency, audio = scipy.io.wavfile.read(ts)
      value = randint(0,5)
      print("value ",value)
      noise_chosen = noise_list[value]
      index = randint(0,800000)
      result = 0.5*audio[:len(audio)] + 0.5*noise_chosen[index:index+len(audio)]
      audio = result
      #print(audio)
      #length = len(audio)
      #next_length = next_power_of_2(length)

      #audio_main = []

      #for i in range (length):
      #  audio_main.append(audio[i])

      #or i in range(next_length - length):
      #  audio_main.append(0)
      audio_main = audio
      #audio_fft = fourier_transform(audio_main)
      #print(audio_fft)
      #print(audio_fft.shape)
      #print(np.fft.fft(audio_main))
      starts_array, specto = spectrogram_make(audio_main,sampling_frequency)
      #print("spectrogram ", len(specto))
      print(specto)
      specto2 = specto.ravel()
      #specto2 = specto2[:10000]
      X_test.append(specto2)
      folder_array = folder.split("/")
      y_test.append(folder_array[6])
      #print(specto)
      #break
      print("count ",count)
      count=count+1
      print("Hello")
    except:
      print("Some error")

import pickle
output4 = open('/content/drive/My Drive/Pickle_X_test2'+ '.pickle', 'wb')
pickle.dump(X_test, output4)
output5 = open('/content/drive/My Drive/Pickle_y_test2'+ '.pickle', 'wb')
pickle.dump(y_test, output5)
output6 = open('/content/drive/My Drive/Pickle_X_train_final2'+ '.pickle', 'wb')
pickle.dump(X_train, output6)

from sklearn import metrics
for i in range(len(X_test)):
  #print(X_train[i].shape)
  if(X_test[i].shape[0] < 15872):
    temp = np.zeros(15872)
    for k in range(X_test[i].shape[0]):
      temp[k] = (X_test[i])[k]
    for k in range(15872 - X_test[i].shape[0]):
      temp[X_test[i].shape[0] + k] = 0
    
    X_test[i] = temp

    print("HELLLOOOO ",X_test[i].shape)
    for y in range (15872):
      if((X_test[i])[y] == float('inf') or (X_test[i])[y] == -float('inf')):
        (X_test[i])[y] = 0
  X_test[i] = np.nan_to_num(X_test[i])
  #X_test[i] = np.array(X_test[i], dtype=np.float64)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

output7 = open('/content/drive/My Drive/Pickle_X_test_final2'+ '.pickle', 'wb')
pickle.dump(X_test, output6)

print("Precision:",metrics.precision_score(y_test, y_pred, average = None))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average = None))

"""References"""
#https://towardsdatascience.com/fast-fourier-transform-937926e591cb
#https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html
#https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
#https://timsainburg.com/python-mel-compression-inversion.html
#https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python