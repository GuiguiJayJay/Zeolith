"""
========================================
Zeolithe's pictures classification draft
========================================
"""
print(__doc__)

import numpy as np
import tensorflow as tf
import cv2

import os
import time
import argparse

import zeolib.preproc as prep
import zeolib.utils as utils
import zeolib.models as mod
import zeolib.settings as GLOB

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
GLOB.init()

# Parse inputs
parser = argparse.ArgumentParser(description='Define important inputs.')
parser.add_argument("--test",
                    type=int,
                    default=0,
                    help="misc tests mode")
                    
FLAGS, unparsed = parser.parse_known_args()

if (GLOB.ML_mode['CNN_Pred']==True) & (GLOB.ML_mode['CNN_Train']==False):
  GLOB.num_pics = {'test': 0}
  GLOB.labels = {'test': 0}
  GLOB.splitsize = 1

########################
# MISC TESTS
########################
#_,_,_ = prep.picprep('DE6117021_21_target2_10.png', '2Faces')


########################
# DATA PRE-PROCESSING
########################
data, labels, bonus, filenames = prep.databuild()


########################
# MACHINE LEARNING PART
########################

if (GLOB.ML_mode['CNN_Pred']==False) | (GLOB.ML_mode['CNN_Train']==True):
  # Create train and validation sets
  trainindex, valindex = utils.setcreator()
  datadict = {'train_data': np.array([data[i] for i in trainindex], dtype=np.float32),
              'train_labels': np.array([labels[i] for i in trainindex], dtype=np.float32),
              'train_bonus': np.array([bonus[i] for i in trainindex], dtype=np.float32),
              'train_names': np.array([filenames[i] for i in trainindex], dtype=str),
              'valid_data': np.array([data[i] for i in valindex], dtype=np.float32),
              'valid_labels': np.array([labels[i] for i in valindex], dtype=np.float32),
              'valid_bonus': np.array([bonus[i] for i in valindex], dtype=np.float32),
              'valid_names': np.array([filenames[i] for i in trainindex], dtype=str) }
else:
  # Create test set
  datadict = {'test_data': np.array([data[i] for i in range(GLOB.num_data)], dtype=np.float32),
              'test_bonus': np.array([bonus[i] for i in range(GLOB.num_data)], dtype=np.float32),
              'test_names': np.array([filenames[i] for i in range(GLOB.num_data)], dtype=str) }

# Run the CNN
mod.run_cnn(init_lr = 0.0001,
            decay_rate = 0.95,
            batchsize = 1,
            epochs = 50,
            droprate = 0.2,
            data = datadict)


########################
# TODO
########################
"""
- create github (edit .gitignore)
- test dropout values and positions
- test L2 regularization: needs to be computed from weights directly
    -> need to change cnn function to calculate and return the L2 loss
    -> need to add the L2 loss to regular loss in cnn_run function
- test another activation function for DNN stack
- implement a stopping criterion depending on validation loss evolution
  (if train loss decreases but validation losses increases for say, 3 consecutive runs,
  then stop learning and go back to configuration 3 runs ago)
    
*Models:
  - write a separate model for your bonus features
      - check a simple SVM (scikit-learn as a separate model in main.py)
      - check a short DNN
      - anyway output a probability normalized please!
  - join the results of the 2 models at the logits steps and divide by 2:
      - do it in tensorflow in a single call of main func (just create another model function)
  - train with this and see if it works
      
"""


