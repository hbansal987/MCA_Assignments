# -*- coding: utf-8 -*-
"""Question2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s8UHRxXcfYAG8-a4MgMywtx2AgLt7U1Y
"""

import cv2
import numpy as np
import time
import sys
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
import glob
import pickle
from pathlib import Path

#print("Hello")

flag=0
path = "/content/drive/My Drive/HW-1/images/*.jpg"
total_time = []
for file in glob.glob(path):
  try:
    flag=flag+1
    #if(flag >= 3123):
    print(flag)
    print(file)
    d= Path(file).stem
    print(d)
    image = cv2.imread(file)
    image = cv2.resize(image,(500,500))

    level = 16
    threshold = 0.02
    initial_sigma = 1.3
    sigma_factor = 1.24

    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    s_time = time.time()

    scale_space,sigma = create_space(gray_image,initial_sigma,sigma_factor,level)

    blobLocation = location(scale_space,sigma,threshold,level)

    gray_image_pixel = []
    image_pixel = []
    for c in blobLocation:
      r = int(np.ceil(sigma[c[2]]*np.sqrt(2)))
      cv2.circle(image,(c[1],c[0]),r, (0,0,255))
      gray_image_pixel.append(gray_image[c[1],c[0]])
      image_pixel.append(image[c[1],c[0]])
      #print("pixel ", image[c[1],c[0]]) 

      #plt.imshow(image)
      #print(len(blobLocation))
    print("Total Time Taken (in seconds): {}".format(time.time() - s_time))

      #outp1 = open('/content/drive/My Drive/PickleFiles_Question2_BlobLocation/'+d+'.pickle', 'wb')
      #pickle.dump(blobLocation,outp1)
      #total_time.append(time.time() - s_time)

      #outp2 = open('/content/drive/My Drive/PickleFiles_Question2_GrayPixel/'+d+'.pickle', 'wb')
      #pickle.dump(gray_image_pixel,outp2)
      #total_time.append(time.time() - s_time)

      #outp3 = open('/content/drive/My Drive/PickleFiles_Question2_ImagePixel/'+d+'.pickle', 'wb')
      #pickle.dump(image_pixel,outp3)
      #print("Hello")
      #total_time.append(time.time() - s_time)

      #flag=flag+1
    if(flag==1):
      break
    #except:
    #print("There was some error")
t=0
for k in total_time:
  t=t+k
print(t)

def create_space(gray_image, initial_sigma,sigma_factor,level):
  height,width = gray_image.shape
  scale_array = np.zeros((height+2,width+2,level))
  sigma = []
  sigma.append(initial_sigma)
  for i in range(1,level):
    sigma.append(sigma[i-1] * sigma_factor)

  for i in range(level):
    kernel = LoG(sigma[i])
    f = kernel.shape[0]
    convolved_image = convolve(gray_image,kernel,height,width,f)
    scale_array[:,:,i] = np.power(convolved_image,2)
  return scale_array,sigma

def LoG(sigma_value):
  k_size = np.round(6*sigma_value)
  if(k_size % 2) == 0:
    k_size = k_size+1
  
  h_size = np.floor(k_size/2)

  a = np.arange(-h_size,h_size+1)
  x , y = np.meshgrid(a,a)

  exponent_term = np.exp(-(np.power(x,2)+np.power(y,2))/(2*np.power(sigma_value,2)))
  value = sys.float_info.epsilon * exponent_term.max()
  for i in exponent_term:
    for j in range (len(i)):
      if(i[j] < value):
        i[j] = 0;

  sum = 0
  for i in exponent_term:
    for j in range (len(i)):
      sum = sum + i[j]
  #print("exponent_term ",sum)
  #print("exponent_term.sum ", exponent_term.sum)
  
  #print("exponent_term.sum ", exponent_term.sum())
  if(sum != 0):
    exponent_term = exponent_term/sum

  kernel = -((np.power(x,2) + np.power(y,2) - (2*np.power(sigma_value,2))) / np.power(sigma_value,2)) * exponent_term
  #print("kernel ",kernel)

  """m = 0
  m2 = 0
  for i in kernel:
      m = m + i

  m2 = m.sum()/len(m)
  
  print("m ", m2)
  print("kernel.mean ", kernel.mean())"""
  kernel = kernel - kernel.mean() ####### Can change
  return kernel

def convolve(gray_image,kernel,height,width,f):
  p = int((f-1)/2)
  image_pad = np.pad(gray_image,p)
  
  V_dim = height+ (2*p) -f +1
  H_dim = width+ (2*p) -f +1

  out_image = np.zeros((V_dim,H_dim))

  c_image = cv2.filter2D(gray_image,-1,kernel)

  #for y in range (p,height+p):
  #  for x in range (p,width+p):
  #    r = padded_img[y - p:p + y + 1, -p + x:p + x + 1]
  #    out_image[y-p,x-p] = (r*kernel).sum #Can be changed

  c_image = np.pad(c_image, (1,1), 'constant')
  #c_image = np.square(c_image)

  #print("c_image ", c_image) 

  return c_image

def location(scale_space,sigma,threshold,level):
  scale_space_copy = np.zeros((scale_space.shape))
  #print("scale_space ", scale_space)
  g=0
  for i in scale_space:
    scale_space_copy[g] = i
    g=g+1
  size = np.shape(scale_space[:,:,0])

  mask =[]
  #for i in range(level):
  #  mask.append(0)
  index = [(1,1),(-1,-1),(-1,1),(1,-1),(1,0),(0,1),(-1,0),(0,-1)]

  for i in range (level):
    a = int(np.ceil(np.sqrt(2)*sigma[i]))
    mask.append(a)

  #print("MASK ", mask)
  blob_location = []

  #def check(l):
  #  counter = True
  #  for v in index:
  #    dx = v[0]
  #    dy = v[1]
  #    if(0<= i + dx < size[0] and 0<= j + dy <size[1]):
  #      if(scale_space[i + dx, j + dy, l] < scale_space[i, j, k]):
  #        counter = True
  #      else:
  #        return False
  #  return True
    
  for k in range (level):
    scale_space_copy[-mask[k]:,-mask[k]:,k] = 0
    scale_space_copy[:mask[k],:mask[k],k] = 0

    b = mask[k]+1

    for i in range(b,size[0]-mask[k]-1):
      for j in range(b,size[1]-mask[k]-1):
        if(scale_space[i,j,k] >= threshold):
          c_max = check(k,index,i,j,size,scale_space,k)
          max1 = True
          max2 = True
          if (-1 < k-1):
            #print("HELLO ", check(k-1))
            max1 = check(k - 1,index,i,j,size,scale_space,k) and (scale_space[i, j, k - 1] < scale_space[i, j, k])

          if (level > k+1):
            max2 = check(k + 1,index,i,j,size,scale_space,k) and (scale_space[i, j, k + 1] < scale_space[i, j, k])

          if(c_max == True and max1 == True and max2 == True):
            scale_space_copy[i,j,k] = 1
            blob_location.append((i,j,k))

  print(blob_location)
  return blob_location

def check(l,index,i,j,size,scale_space,k):
    counter = True
    for v in index:
      dx = v[0]
      dy = v[1]
      #print("dx ",dx)
      #print("i ", i)
      if(0<= i + dx < size[0] and 0<= j + dy <size[1]):
        if(scale_space[i + dx, j + dy, l] < scale_space[i, j, k]):
          counter = True
        else:
          return False
    return True

"""DATA_PATH = "/content/drive/My Drive/PickleFiles_Question2_BlobLocation"
infile = open(DATA_PATH+'/oxford_002627.pickle','rb')
best_model2 = pickle.load(infile)
print(best_model2)

from google.colab import drive
import cv2 
import glob
import numpy as np
import pickle
from numpy import linalg as LA

with open('/content/drive/My Drive/HW-1/train/query/all_souls_1_query.txt', 'r+') as f:
  a=f.read()

a = a.split(" ")
a = a[0]
a = a[5:]
print(a)

path = "/content/drive/My Drive/HW-1/images/" + a + ".jpg"
img = cv2.imread(path)
#print("image ",img)

g = blob(img)

path_pickle = '/content/drive/My Drive/PickleFiles_Question2_GrayPixel/*.pickle'
dicti = {}
flag=0
all_gray_pixels = []
for file in glob.glob(path_pickle):
  try:
    flag=flag+1
    print(flag)
    #print("file ", file)
    infile = open(file, 'rb')
    best_model = pickle.load(infile)
    #print ("temp_pickle ", best_model)  
    all_gray_pixels.append(best_model)
    t= []
    mi=0
    if(len(g) < len(best_model)):
      mi= len(g)
    else:
      mi = len(best_model)
    for i in range(mi):
      t.append(g[i] - best_model[i])
    #t = corr[0] - best_model[0]
    #print("t ", t)
    value = LA.norm(t)
    #print("value ", value)
    dicti[file] = value
  except:
    print("NOT EMPTY")
print("dict ", dicti)
print("Hello ",sorted(dicti.items(), key = 
             lambda kv:(kv[1], kv[0])))

import cv2
import numpy as np
import time
import sys
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
import glob
import pickle
from pathlib import Path

def blob(image):
  level = 16
  threshold = 0.02
  initial_sigma = 1.3
  sigma_factor = 1.24

  image = cv2.resize(image,(500,500))

  gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  gray_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  s_time = time.time()

  scale_space,sigma = create_space(gray_image,initial_sigma,sigma_factor,level)

  blobLocation = location(scale_space,sigma,threshold,level)
  gray_image_pixel = []
  image_pixel = []
  for c in blobLocation:
    r = int(np.ceil(sigma[c[2]]*np.sqrt(2)))
    cv2.circle(image,(c[1],c[0]),r, (0,0,255))
    #print(gray_image[c[1],c[0]])
    gray_image_pixel.append(gray_image[c[1],c[0]])
    image_pixel.append(image[c[1],c[0]])
  print("Total Time Taken (in seconds): {}".format(time.time() - s_time))

  return (gray_image_pixel)
"""