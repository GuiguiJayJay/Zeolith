import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
from os.path import isfile, join

import zeolib.utils as utils
import zeolib.settings as GLOB


############################################
def databuild():
  """ Build data/label arrays.
  
  Parameters
  ==========
    
  Returns
  =======
  data: python list
    Python list of flatten arrays (the pictures).
        
  labels: python list
    Python list holding the data label.
    
  bonus: python list
    Python list holding the 4 max blank areas found on pic with 2 v/h slices.
    
  filenames: python list
    Python list holding the names of files (monitoring purposes).
  """

  start_time_tot = time.time()

  height = []
  width = []
  picnames = {}
  data = []
  labels = []
  bonus = []
  filenames = []
  
  # Store filenames and find max size for pictures
  if GLOB.maxfinder == True:             
    for item in GLOB.num_pics.keys():
      picnames[item] = []
      for name in os.listdir(GLOB.directories[item]):
        if isfile(join(GLOB.directories[item], name)):
          picnames[item].append(name)
          img = cv2.imread(GLOB.directories[item] + name, 0)
          height.append(img.shape[0])
          width.append(img.shape[1])
    
    GLOB.maxheight = max(height)
    GLOB.maxwidth = max(width)
    if GLOB.MONITOR==True: print('Max pic (height, width) are (%d, %d)' % (max(height), max(width)) )
  
  # Prepare pictures for work
  for item in GLOB.num_pics.keys():
    for name in os.listdir(GLOB.directories[item]):
      if isfile(join(GLOB.directories[item], name)):
        picdata, piclabel, faces = picprep(name=name, item=item)
        data.append(picdata)
        labels.append(piclabel)
        bonus.append(faces)
        filenames.append(name)
        
        # Number of elements per category
        GLOB.num_pics[item] += 1
        
    # Total number of elements
    GLOB.num_data += GLOB.num_pics[item]
    
  if GLOB.TIMER==True: print("-> Pre-processing = %0.2f s \n" % (time.time()-start_time_tot) )
  
  return data, labels, bonus, filenames


############################################
def picprep(name='', item=''):
  """ Preprocess the input pictures.
  
  Parameters
  ==========
  name: str
    Name of the picture.
    
  item:
    Category of the picture (1Face, 2Faces etc...).
    
  Returns
  =======
  data: python list
    Python list of flatten arrays (the pictures).
        
  labels: python list
    Python list holding the data label.
        
  faces: np.array (1D)
    A 16 element array holding the 4 max blank areas for the 2 row/col
    slices. Store a zero if a given slice has less than 4 blank areas.
  """
  
  # Open pic and resize
  img = cv2.imread(GLOB.directories[item] + name, 0)
  
  # Smooth the pic to remove local noises (averaging 3x3 areas)
  kernel1 = np.ones((3,3),np.float32)/9
  imout = cv2.filter2D(img,-1,kernel1)
  
  # Calculate pixel intensities Laplacian (and rescale to 0-255)
  laplacian = cv2.Laplacian(src=imout, ksize=5, ddepth=cv2.CV_8S)
  laplacian = (laplacian + 128).astype(np.uint8)

  # Bilateral filter to smooth faces
  blur = cv2.bilateralFilter(laplacian,9,150,100)

  # Adapt. Tresh.
  thresh = cv2.adaptiveThreshold(blur, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,9,20)
  
  # Downsize picture
  imout1 = cv2.resize(thresh, (GLOB.maxheight,GLOB.maxwidth), interpolation=cv2.INTER_CUBIC)
  
  #print(name)
  #plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original')
  #plt.xticks([]), plt.yticks([])
  #plt.subplot(232), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
  #plt.xticks([]), plt.yticks([])
  #plt.subplot(233), plt.imshow(blur, cmap='gray'), plt.title('Bil. Filter 9-150-100')
  #plt.xticks([]), plt.yticks([])
  #plt.subplot(234), plt.imshow(thresh, cmap='gray'), plt.title('Adapt.Thresh 9-20')
  #plt.xticks([]), plt.yticks([])
  #plt.subplot(235), plt.imshow(imout1, cmap='gray'), plt.title('Downsize 128')
  #plt.xticks([]), plt.yticks([])
  #plt.show()
  
  faces = None
  if GLOB.ML_mode['Sampler'] == True:
    #utils.plotter(img=imout1)
    faces = utils.sampler(img=imout1)
    #plt.imshow(imout1, cmap='gray'), plt.title('Downsize 128')
    #plt.xticks([]), plt.yticks([])
    #plt.show()
  
  # Fills data into a list of flatten arrays and associated labels
  picdata = imout1.flatten()
  piclabel = GLOB.labels[item]
  
  return picdata, piclabel, faces
  



