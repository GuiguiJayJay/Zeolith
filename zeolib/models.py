import numpy as np
import tensorflow as tf
import time

import zeolib.settings as GLOB
import zeolib.utils as utils


############################################
def cnn(X, Xbonus, initializer=None, keep_prob=None):
  """Model function for CNN (from tutorial).
    
  Parameters
  ==========
  X
    The input features reshaped as a 1D array per batch elements.
    
  Xbonus
    The bonus features from blank detector (16 elements as 1D array).

  initializer: tf initializer
    A particluar initializer to set starting weights of each layers.
    
  keep_prob: tf placeholder (1 element)
    Probability to keep a neuron in a dense layer (for dropout technique).
    
  Returns
  =======
  None
  """

  # Input Layer
  input_layer = tf.reshape(X, [-1, GLOB.maxwidth, GLOB.maxheight, 1])
  
  # Convolutional Layer #1 (output=[batchsize,maxwidth,maxheight,filters])
  conv1 = tf.layers.conv2d(inputs=input_layer,
                           filters=50,
                           kernel_size=[8, 8],
                           strides=(2, 2),
                           padding="same",
                           activation=tf.nn.relu)
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                  pool_size=[2, 2],
                                  strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(inputs=pool1,
                           filters=100,
                           kernel_size=[4, 4],
                           padding="same",
                           activation=tf.nn.relu)
  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                  pool_size=[2, 2],
                                  strides=2)
  
  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(inputs=pool2,
                           filters=200,
                           kernel_size=[3, 3],
                           padding="same",
                           activation=tf.nn.relu)
  
  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(inputs=conv3,
                           filters=200,
                           kernel_size=[3, 3],
                           padding="same",
                           activation=tf.nn.relu)

  # Dense Layer #1
  size = int(GLOB.maxwidth/8) * int(GLOB.maxheight/8) * 200
  output_flat = tf.reshape(conv4, [-1, size])
  dense1 = tf.layers.dense(inputs=output_flat,
                           units=2000,
                           activation=tf.nn.relu)
  dropout1 = tf.nn.dropout(dense1, keep_prob)
  
  # Dense Layer #2
  bonusfeat = tf.concat([dropout1, Xbonus], 1)
  dense2 = tf.layers.dense(inputs=bonusfeat,
                           units=200,
                           activation=tf.nn.relu)
  #dense2 = tf.layers.dense(inputs=dropout1,
                           #units=200,
                           #activation=tf.nn.relu)
  dropout2 = tf.nn.dropout(dense2, keep_prob)
  
  # Dense Layer #3
  dense3 = tf.layers.dense(inputs=dropout2,
                           units=200,
                           activation=tf.nn.relu)
  dropout3 = tf.nn.dropout(dense3, keep_prob)
  
  # Dense Layer #4
  dense4 = tf.layers.dense(inputs=dropout3,
                           units=100,
                           activation=tf.nn.relu)
  dropout4 = tf.nn.dropout(dense3, keep_prob)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout4, units=3)

  return logits
  

############################################
def run_cnn(init_lr=0.1,
            decay_rate=0.95,
            batchsize=1,
            epochs=1,
            droprate=1,
            data={}):          
  """ Runs the specified Neural-Network.
  
  Parameters
  ==========
  init_lr: float
    The starting learning rate.
    
  decay_rate: float
    The decay rate.
    
  batchsize: int
    The size of each data batch.
    
  epochs: int
    The number of epochs (full data cycle).
    
  droprate: float
    The fraction of neurons to be kept while training.
      
  data: python dict
    Dictionary holding the dataset names and associated data.

  Returns
  =======
  None
  """
    
  # Misc. var definitions
  learning_rate = tf.placeholder(tf.float32, shape=[])
  steps = int(GLOB.splitsize*GLOB.num_data/batchsize) 
  initializer = tf.random_normal_initializer(mean=0.0, stddev=0.5, dtype=tf.float32)
  droprate = 1 - droprate # converts droprate into keep prob
  keep_prob = tf.placeholder(tf.float32)
   
  featsize = GLOB.maxheight * GLOB.maxwidth
  X = tf.placeholder('float32', [None, featsize])
  Xbonus = tf.placeholder('float32', [None, 16])
  labels = tf.placeholder('int32', [None])
  
  logits = cnn(X, Xbonus, initializer, keep_prob)
  
  # Loss
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
  
  # Accuracy
  predclasses = tf.cast(tf.argmax(input=logits, axis=1), tf.int32) # just a shortcut
  predictions = {"classes": predclasses,
                 "classes-OH": tf.one_hot(indices=predclasses, depth=3),
                 "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}              
  correct = tf.equal(predictions["classes"], labels)
  accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
  # Optimization
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

  # Create a Saver to save NN weights
  saver = tf.train.Saver()

  # Add variables to be written to a summary
  tf.summary.scalar('loss', loss)
  summary = tf.summary.merge_all()
  
  # Create a Python dict to hold various TensorFlow objects, and Training parameters
  tfdict = {'loss': loss,
            'accuracy': accuracy,
            'predclasses': predclasses,
            'predclassesOH': predictions['classes-OH'],
            'optimizer': optimizer,
            'keep_prob': keep_prob,
            'summary': summary,
            'saver': saver,
            'X': X,
            'Xbonus': Xbonus,
            'labels': labels,
            'labelsOH': onehot_labels,
            'learning_rate': learning_rate}
            
  params = {'init_lr': init_lr,
            'decay_rate': decay_rate,
            'batchsize': batchsize,
            'epochs': epochs,
            'droprate': droprate,
            'steps': steps}
  
  # train the CNN to generate weights
  if GLOB.ML_mode['CNN_Train'] == True:
    print('Start training CNN')
    CNN_training(params = params,
                data = data,
                tfobjects = tfdict)
  # Load weights from disk
  else:
    print('Use weights from the disk')
    CNN_load(params = params,
             data = data,
             tfobjects = tfdict)
                                                                 
  return None
                                    
                                  
############################################
def CNN_training(params={},
                 data={},
                 tfobjects={}):
  """Runs the previously defined CNN.
    
  Parameters
  ==========
  params: python dict
    Dictionnary holding training parameters and their names
      'init_lr' (float): The starting learning rate.
      'decay_rate' (float): The decay rate.
      'batchsize' (int): The size of each data batch.
      'epochs' (int): The number of epochs (full data cycle).
      'droprate' (float): The fraction of neurons to be kept for training.
      'steps'(int): The number of steps per epoch.
      
  data: python dict
    Dictionary holding the dataset name and associated data.

  tfobjects: python dict
    Dictionary holding TensorFlow objects and their names.
        
  Returns
  =======
  None
  """    

  start_time_tot = time.time()
  init = tf.global_variables_initializer()
  
  # Open output file
  oufile = open('plots/output.txt', 'w+')

  with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('train/', sess.graph)
    prevloss = 100
    cumulativeloss = 0
        
    for j in range(0,params['epochs']):
      print('Learning rate: %0.4e' % params['init_lr'])
      start_time_part = time.time()
      loss = 0
      
      for i in range(0,params['steps']):
        # Adapt size of last batch
        stop = i + params['batchsize']
        if stop > int(GLOB.num_data * GLOB.splitsize):
          stop = int(GLOB.num_data * GLOB.splitsize)
          
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, summary = sess.run([tfobjects['optimizer'],
                                  tfobjects['loss'],
                                  tfobjects['summary']],
                                 feed_dict={tfobjects['X']: data['train_data'][i:stop],
                                            tfobjects['labels']: data['train_labels'][i:stop],
                                            tfobjects['Xbonus']: data['train_bonus'][i:stop],
                                            tfobjects['keep_prob']: params['droprate'],
                                            tfobjects['learning_rate']: params['init_lr']})
        summary_writer.add_summary(summary, i*(j+1))
        cumulativeloss += l
      
      # Metrics and debug hand-made style
      trainloss, trainacc = utils.metrics(tfobjects = tfobjects,
                                          X = data['train_data'],
                                          Xbonus = data['train_bonus'],
                                          labels = data['train_labels'],
                                          sess = sess,
                                          key = 'train')
      validloss, validacc = utils.metrics(tfobjects = tfobjects,
                                          X = data['valid_data'],
                                          Xbonus = data['valid_bonus'],
                                          labels = data['valid_labels'],
                                          sess = sess,
                                          key = 'test')
                        
      cumulativeloss = cumulativeloss/len(data['train_labels'])
      print('Epoch cumulated training loss: %0.4f' % cumulativeloss)
      
      oufile.write('%d %0.4f %0.2f %0 4f %0.2f %0.4f' % (j+1,
                                                         trainloss,
                                                         trainacc,
                                                         validloss,
                                                         validacc,
                                                         cumulativeloss))
      oufile.write('\n')
      
                    
      # Adjust learning if loss progress by less than few %
      if (cumulativeloss/prevloss) > 0.95:
          params['init_lr'] = params['init_lr']*params['decay_rate']
      prevloss = cumulativeloss
        
      if GLOB.TIMER==True: print("-> Epoch %d training time = %0.2f s \n" % (j, time.time()-start_time_part) )
      
    # Save the NN weights
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    tfobjects['saver'].save(sess, 'weights/Ref05-full/model_weights', global_step=i)
    
    if GLOB.TIMER==True: print("-> CNN full training time = %0.2f s \n" % (time.time()-start_time_tot) )
    
  # Reset the graph or the loop over the datasets will not work
  tf.reset_default_graph()
  # End session to avoid any (more) memory problems and close output file
  sess.close()
  oufile.close()

  return None
      
      
  
############################################
def CNN_load(tfobjects={},
             data={},
             params={}): 
  """ Use saved Neural Network coefficients to make predictions.
  
  Parameters
  ==========  
  tfobjects: python dict
    Dictionary holding TensorFlow objects and their names.
    
  data: python dict
    Dictionary holding the feature names and associated data.
    
  params: python dict
    Dictionnary holding training parameters and their names
      'init_lr' (float): The starting learning rate.
      'decay_rate' (float): The decay rate.
      'batchsize' (int): The size of each data batch.
      'epochs' (int): The number of epochs (full data cycle).
      'steps'(int): The number of steps per epoch.
    
  Returns
  =======
  None
  """
  
  start_time_tot = time.time()
  init = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init)
    
    # Load the NN weights
    datapoints = '281' # 196
    filename =  'weights/Ref05-full/model_weights-' + datapoints
    tfobjects['saver'].restore(sess, filename)
    
    # Compute total accuracy on given set
    if GLOB.ML_mode['CNN_Pred'] == False:
      print('Calculating accuracy ....')
      loss, dacc = utils.metrics(tfobjects = tfobjects,
                                 X = data['train_data'],
                                 Xbonus = data['train_bonus'],
                                 labels = data['train_labels'],
                                 sess = sess,
                                 key = 'test')
      
      print("-> Neural Network running time = %0.2f s \n" % (time.time()-start_time_tot) )
      
    # Make predictions on test set
    else:
      print('Calculating predictions ....')
      predictions = []
      for i in range(GLOB.num_data):
        stop = i + 1
        predclasses = tfobjects['predclasses'].eval({tfobjects['X']: data['test_data'][i:stop],
                                                     tfobjects['Xbonus']: data['test_bonus'][i:stop],
                                                     tfobjects['keep_prob']: 1.0})        
        print(data['test_names'][i], predclasses)
      
    print("-> Neural Network running time = %0.2f s \n" % (time.time()-start_time_tot) )
    

  # Reset the graph or the loop over the datasets will not work
  tf.reset_default_graph()
  # End session to avoid any (more) memory problems
  sess.close()

  return None


