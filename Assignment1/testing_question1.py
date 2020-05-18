# -*- coding: utf-8 -*-
"""Testing_Question1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J6qdI6nfD0DajQuJAuTYTGoZBgHwvXvL
"""

import numpy as np
import cv2
import math

def clusterForm(image):
  for i in image:
    for j in i:
      #print("j ", j)
      j[0] = changeColor(j[0]) 
      j[1] = changeColor(j[1])
      j[2] = changeColor(j[2])
      #print("j",j)
  return(image)

def changeColor(v):
  if(v<64):
    return 0
  if(v < 128):
    return 1
  if(v<192):
    return 2
  if(v<256):
    return 3

def autocorrelogram(image):
  
  #print("1111111111 " ,image.shape)
  image = cv2.resize(image,(500,500))
  """Z = image.reshape((-1, 3)).astype(np.float32)
  #print("2222222222 " ,Z.shape)
  K = 64

  #ret, label, centre = cv2.kmeans(Z,K,bestLabels= None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,flags = cv2.KMEANS_PP_CENTERS)
  ret, label, centre = cv2.kmeans(data=Z, K=64, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 4, 1.0),
                                    attempts=4, flags=cv2.KMEANS_PP_CENTERS,
                                    bestLabels=None)
  #centre = np.uint8(centre)
  x = label.flatten()
  res = centre[x]
  res2 = res.reshape((image.shape))

  print("res ",res)
  print("res2", res2)
  print("res2.shape " , res2.shape)
  print("res.shape ", res.shape)

  print("HIMANSHU ", res[4000])
  unique_res = unique(np.array(res))

  print("unique_res" , unique_res)

  print("unique_res.shape ", unique_res.shape)

  #print("unique_res ", unique_res)

  print("res2.shape ", res2.shape)"""

  D = [1,3,5,7]
  img = clusterForm(image)
  #print("img ", img[250,250,2])
  bins = []
  
  for i in range (4):
    for j in range(4):
      for k in range (4):
        temp = []
        temp.append(0)
        temp.append(0)
        temp.append(0)
        temp[0] = i
        temp[1] = j
        temp[2] = k
        bins.append(temp)
  #print("bins ", bins)
  result = correlogram(img,bins,D)

  return result

def correlogram(image,Cm,D):

  #print("CORRELOGRAM 1")
  X,Y,t = image.shape
  colorspercent = []

  for d in (D):

    color_array = []
    color_count =0
    for c in range (len(Cm)):
      color_array.append(0)

    #print("d " , d)
    for x in range(0, X, int(round(X / 5))):
      for y in range(0, Y, int(round(Y / 5))):
        #print("x ",x)
        #print("y ",y)
        Ci = image[x][y]
        Cn = getNeighbour(X,Y,x,y,d)

        for j in Cn:
          Cj = image[j[0]][j[1]]

          for m in range(len(Cm)):
            if((len(Cm[m]) == len (Cj)) and(len(Cm[m]) == len(Ci))):
              for s in range (len(Cm[m])):
                #print("22222222222")
                if(((Cm[m])[s] == Cj[s]) and ((Cm[m])[s] == Ci[s])):
                  flag=1
                else:
                  flag=0
                if(flag==1):
                  color_count = color_count+1;
                  color_array[m] = color_array[m] + 1
            

    #print("Bye")
    for i in range (len(color_array)):
      color_array[i] = color_array[i]/color_count

    colorspercent.append(color_array)

  return colorspercent

def getNeighbour(X,Y,x,y,d):
  p1 = (x+d,y+d)
  p2 = (x-d,y-d)
  p3 = (x+d,y)
  p4 = (x-d,y)
  p5 = (x,y+d)
  p6 = (x,y-d)
  p7 = (x+d,y-d)
  p8 = (x-d,y+d)

  p = [p1,p2,p3,p4,p5,p6,p7,p8]

  C = []
  for i in p:
    if(i[0] >= 0 and i[0] <X):
      if([i[1] >= 0 and i[1] <Y]):
        C.append(i)

  return C

"""import glob
import pickle
import time
import cv2
a= cv2.imread("/content/drive/My Drive/HW-1/images/all_souls_000000.jpg")
ans = autocorrelogram(a)
print("ans ", ans)

import glob
import pickle
import time
from pathlib import Path

path = "/content/drive/My Drive/HW-1/images/*.jpg"
flag=0
ans_final = []
t=[]
for file in glob.glob(path):
  try:
    if(flag > 3140):
      print(flag)
      print(file)
      #d = file.spilt(".")
      #print(d[0])
      d = Path(file).stem
      print(d)
      start_time = time.time()
      a= cv2.imread(file)
      ans = autocorrelogram(a)
      print(ans)
      
      print("Total Time Taken (in seconds): {}".format(time.time() - start_time))
      t.append(time.time() - start_time)
      output = open('/content/drive/My Drive/PickleFile_Question1/'+ d + '.pickle', 'wb')
      pickle.dump(ans, output)
      #if(flag==5):
      #  break
    flag=flag+1
  except:
    print("some error has occured")

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

corr = autocorrelogram(img)
print(corr)

path_pickle = '/content/drive/My Drive/PickleFile_Question1/*.pickle'
dicti = {}
for file in glob.glob(path_pickle):
  try:
    print("file ", file)
    infile = open(file, 'rb')
    best_model = pickle.load(infile)
    #print ("temp_pickle ", best_model)  
    t= []
    for i in range(len(corr[0])):
      t.append((corr[0])[i] - (best_model[0])[i])
    #t = corr[0] - best_model[0]
    #print("t ", t)
    value = LA.norm(t)
    print("value ", value)
    dicti[file] = value
  except:
    print("NOT EMPTY")
print("dict ", dicti)
print("Hello ",sorted(dicti.items(), key = 
             lambda kv:(kv[1], kv[0]))) 

with open('/content/drive/My Drive/HW-1/train/ground_truth/all_souls_1_good.txt','r+') as f_good:
  good = f_good.read()
good = good.split("\n")
print("good ", good[0])

with open('/content/drive/My Drive/HW-1/train/ground_truth/all_souls_1_junk.txt','r+') as f_junk:
  junk = f_junk.read()
junk = junk.split("\n")
print("junk ", junk[0])

with open('/content/drive/My Drive/HW-1/train/ground_truth/all_souls_1_ok.txt','r+') as f_ok:
  ok = f_ok.read()
ok = ok.split("\n")
print("ok ",ok[0])
"""

from google.colab import drive
import cv2 
import glob
import numpy as np
import pickle
from numpy import linalg as LA

precision_array = []
recall_array = []
F1_array = []

good_array = []
ok_array = []
junk_array = []

tempo = '/content/drive/My Drive/HW-1/train/query/*.txt'
for query_file in glob.glob(tempo):
  text_file = query_file
  text_file_name = text_file.split("/")
  text_file_name = text_file_name[-1]
  text_file_name = text_file_name[:-9]
  print(text_file_name)
  with open(text_file, 'r+') as f:
    a=f.read()

  a = a.split(" ")
  a = a[0]
  a = a[5:]
  print(a)

  path = "/content/drive/My Drive/HW-1/images/" + a + ".jpg"
  img = cv2.imread(path)
  corr = autocorrelogram(img)
  print(corr)

  infile = open('/content/drive/My Drive/All_PickleFiles_Question1/Features_question1.pickle','rb')
  best_model2 = pickle.load(infile)
  print(len(best_model2))

  diff_values = {}
  for i in best_model2:
    temp = []
    for j in range (len((best_model2[i])[0])):
      temp.append((corr[0])[j] - ((best_model2[i])[0])[j])
    temp = LA.norm(temp)
    diff_values[i] = temp

  print("diff_values ", diff_values)
  print("Hello ",sorted(diff_values.items(), key = 
             lambda kv:(kv[1], kv[0])))
  sorted_values = sorted(diff_values.items(), key = lambda kv:(kv[1], kv[0]))

  with open('/content/drive/My Drive/HW-1/train/ground_truth/' +text_file_name+ 'good.txt','r+') as f_good:
    good = f_good.read()
  good = good.split("\n")

  with open('/content/drive/My Drive/HW-1/train/ground_truth/' +text_file_name+ 'junk.txt','r+') as f_junk:
    junk = f_junk.read()
  junk = junk.split("\n")

  with open('/content/drive/My Drive/HW-1/train/ground_truth/' +text_file_name+ 'ok.txt','r+') as f_ok:
    ok = f_ok.read()
  ok = ok.split("\n")

  total_images_retrieved = 100
  total_relevant_images = len(good) + len(junk) + len(ok)
  print("total_relevant_images ", total_relevant_images)

  total_images_retrieved_correct = 0 
  images_retrived_correct_good = 0
  images_retrived_correct_ok = 0
  images_retrived_correct_junk = 0
  count = 0
  for i in sorted_values:
    i = i[0]
    i = i.split("/")
    i = i[-1]
    #print(i)
    i = i[:-7]
    #print(i)
    if(count < 100):
      for k in good:
        if(i == k):
          #print(count)
          images_retrived_correct_good = images_retrived_correct_good+1
          total_images_retrieved_correct = total_images_retrieved_correct +1 
      for k in ok:
        if(i == k):
          images_retrived_correct_ok = images_retrived_correct_ok+1
          total_images_retrieved_correct = total_images_retrieved_correct +1
      for k in junk:
        if(i == k):
          images_retrived_correct_junk = images_retrived_correct_junk+1
          total_images_retrieved_correct = total_images_retrieved_correct +1
      count = count + 1

  precision = total_images_retrieved_correct/total_images_retrieved
  recall = total_images_retrieved_correct/total_relevant_images
  F1 = 2/((1/precision)+(1/recall))

  print("Precision= ", precision)
  print("Recall= ", recall)
  print("F1= " , F1)

  precision_array.append(precision)
  recall_array.append(recall)
  F1_array.append(F1)

  percentage_of_good_retrived = (images_retrived_correct_good/len(good))*100
  percentage_of_ok_retrived = (images_retrived_correct_ok/len(ok))*100
  percentage_of_junk_retrived = (images_retrived_correct_junk/len(junk))*100

  print("Percentage of good ", percentage_of_good_retrived)
  print("Percentage of ok ", percentage_of_ok_retrived)
  print("Percentage of junk ", percentage_of_junk_retrived)

  good_array.append(percentage_of_good_retrived)
  ok_array.append(percentage_of_ok_retrived)
  junk_array.append(percentage_of_junk_retrived)

print("Max Precision= ", max(precision_array))
print("Min Precision= ", min(precision_array))
print("Average Precision= ", (sum(precision_array)/len(precision_array)))

print("Max recall= ", max(recall_array))
print("Min recall= ", min(recall_array))
print("Average recall= ", (sum(recall_array)/len(recall_array)))

print("Max F1= ", max(F1_array))
print("Min F1= ", min(F1_array))
print("Average F1= ", (sum(F1_array)/len(F1_array)))

print("Average percentage good= ", (sum(good_array)/len(good_array)))
print("Average percentage ok= ", (sum(ok_array)/len(ok_array)))
print("Average percentage junk= ", (sum(junk_array)/len(junk_array)))

print("Max good= ", max(good_array))
print("Min good= ", min(good_array))

print("Max ok= ", max(ok_array))
print("Min ok= ", min(ok_array))

print("Max junk= ", max(junk_array))
print("Min junk= ", min(junk_array))

"""print(percentage_of_good_retrived)"""