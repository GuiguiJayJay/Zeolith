import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import time

import zeolib.settings as GLOB


############################################
def plotter(img=None):
  """ Plot slices of picture (+-10th row/col around the central row).
  
  Parameters
  ==========
  img: np.array(2-3D)
    The picture as an array.
    
  Returns
  =======
  None
  """
  
  centralrow = int(img.shape[0]/2)
  samples = 10

  f, axarr = plt.subplots(3, 2,figsize=(14, 14))
  axarr[0,0].plot(img[centralrow - samples], color='b', alpha=0.75);
  axarr[0,0].set_title('row = %d' % (centralrow - samples))
  axarr[0,1].plot(img[centralrow + samples], color='b', alpha=0.75);
  axarr[0,1].set_title('row = %d' % (centralrow + samples))
  axarr[1,0].plot(img[:,centralrow - samples], color='b', alpha=0.75);
  axarr[1,0].set_title('col = %d' % (centralrow - samples))
  axarr[1,1].plot(img[:,centralrow + samples], color='b', alpha=0.75);
  axarr[1,1].set_title('col = %d' % (centralrow + samples))
  axarr[2,0].imshow(img, cmap='gray')
  axarr[2,0].set_title('Downsize 128')
  plt.show()

  return None
  
  
############################################
def sampler(img=None):
  """ Sample few rows/columns from the pre-processed picture.
  
  Parameters
  ==========
  img: np.array (2-3D)
    numpy array holding the pre-processed picture.
    
  Returns
  =======
  faces: np.array (1D)
    A 16 element array holding the 4 max blank areas for the 2 row/col
    slices. Store a zero if a given slice has less than 4 blank areas.
  """
  
  centralrow = int(img.shape[0]/2)
  samples = 10 # dist. from centre to read row/column
  tempfaces = []

  facesrow1 = facefinder(img[centralrow - samples])
  tempfaces.append(facesrow1)
  facesrow2 = facefinder(img[centralrow + samples])
  tempfaces.append(facesrow2)
  facescol1 = facefinder(img[:,centralrow - samples])
  tempfaces.append(facescol1)
  facescol2 = facefinder(img[:,centralrow + samples])
  tempfaces.append(facescol2)
  
  faces = np.hstack(tempfaces)

  return faces
  

############################################
def facefinder(array=None):
  """ Sample few rows/columns from the pre-processed picture. Extract blank areas and
  edges depending on certain conditions. Derive the number of faces from it.
  
  Parameters
  ==========
  array: np.array (1D)
    numpy array holding a given row/column from the picture.
    
  Returns
  =======
  maxblank: np.array(1D)
    The 4 biggest blank sizes. If less, put zeros to complete the array.
  """
  
  # Build a mask of black pixels
  blackmask = []
  for i in range(GLOB.maxheight):
    if array[i] < 50:
      blackmask.append(i)
      
  # Use the mask to determine edges and blank areas
  edges = []
  blank = []
  tempedge = 0
  tempblank = 0
  for i in range(len(blackmask)-1):
    if blackmask[i+1] - blackmask[i] == 1:
      tempedge += 1
      # If edge found and blank not empty, store the blank length
      if (tempblank>0) & (tempedge>=GLOB.TreshEdge-1):
        blank.append(tempblank)
        tempblank = 0
    else:
      tempblank += blackmask[i+1] - blackmask[i]
      # If blank found and edge not empty, store edge position
      if tempedge>=GLOB.TreshEdge-1: 
        edges.append(blackmask[i])
      tempedge = 0

  # Fill the output vector
  maxblank = np.zeros(shape=[4], dtype='float32')
  blank.sort()
  for i in range(len(maxblank)):
    if i<len(blank):
      maxblank[i] = blank[len(blank)-1-i]/GLOB.maxheight
    else:
      maxblank[i] = 0
      
  return maxblank


############################################
def setcreator():
  """ Creates array of index from data to be sent for training/testing.
  Has to be modified to split equally each category.
  
  Parameters
  ==========
  None
    
  Returns
  =======
  trainindex: np.array
    1D array holding data index in random order for training set.
    
  valindex: np.array
    1D array holding data index in random order for validation set.
  """
    
  perm = np.random.permutation(GLOB.num_data) # random permutation of data
  trainindex = perm[:int(GLOB.splitsize*GLOB.num_data)]
  valindex = perm[int(GLOB.splitsize*GLOB.num_data):]

  return trainindex, valindex
  
  
############################################
def metrics(tfobjects={},
            X=None,
            Xbonus=None,
            labels=None,
            sess=None,
            key=None): 
  """ Calculate various metrics the SAFE WAY (long, but sure!).
  
  Parameters
  ==========  
  tfobjects: python dict
    Dictionary holding TensorFlow objects and their names.
    
  X: np.array (2D)
    Array holding the data.
    
  Xbonus: np.array (1D)
    Array holding the bonus data (blank area sizes).
    
  labels: np.array (1D)
    Array holding the corresponding labels.
  
  sess: tf session
    The tensorflow session the metrics function is ran from.
  
  key: str
    Indicate wether we fed training or validation data. For info display only.
    
  Returns
  =======
  totloss: float
    The total loss from the input dataset.
    
  totacc: float
    The total accuracy from the input dataset.
  """
  
  multiacc = [0,0,0]
  cl_elements = [0,0,0]
  totloss = 0
  for k in range(len(labels)):
    predcl, predclOH, lbl, lblOH, loss = sess.run([tfobjects['predclasses'],
                                                   tfobjects['predclassesOH'],
                                                   tfobjects['labels'],
                                                   tfobjects['labelsOH'],
                                                   tfobjects['loss']],
                                                  feed_dict={tfobjects['X']: X[k:k+1],
                                                             tfobjects['Xbonus']: Xbonus[k:k+1],
                                                             tfobjects['keep_prob']: 1.0,
                                                             tfobjects['labels']: labels[k:k+1]})
    cl_elements[lbl[0]] += 1
    totloss += loss
    if predcl[0] == lbl[0]:
      multiacc[lbl[0]] += 1
    
    # Check the labels array values (pred and defined)
    #print(lbl, lblOH, predcl, predclOH)
  
  # Total accuracy
  totacc = 100*(sum(multiacc) / sum(cl_elements))
  print('Total %s accuracy: %0.2f %%'% (key, totacc) )
  
  # Partial (class-wise) accuracy
  for k in range(0,3):
    multiacc[k] = multiacc[k] / cl_elements[k]
  print('Partial %s accuracy: [%0.2f %%, %0.2f %%, %0.2f %%]' % (key,
                                                                 100*multiacc[0],
                                                                 100*multiacc[1],
                                                                 100*multiacc[2]))
  
  # Total loss
  totloss = totloss/len(labels)
  print('Total %s loss: %0.4f' % (key, totloss) )
  
  # Write results in ouput file (overwrite with each subsequent runs, beware!)
  

  return totloss, totacc


