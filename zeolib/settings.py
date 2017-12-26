
def init():
  """ All the running options switches are defined here,
  as well as the most commonly used variables, and some global variables.
  All of them have been set to global for possible further developpments.
  
  Parameters
  ==========
  None
  
  Returns
  =======
  None
  """
  ########################
  # OPTIONS
  ########################
  # Debugging tools
  global TIMER # displays time of every major step
  TIMER = True
  global MONITOR # displays monitoring infos
  MONITOR = False
  
  global directories
  directories = {'1Face': 'data/1Face/',
                 '2Faces': 'data/2Faces/',
                 '3Faces': 'data/3Faces/',
                 'test': 'data/test/'}
                 
  # Opt. swicthes
  global maxfinder # to find the max dim. amongst the pictures
  maxfinder = False
  global ML_mode
  ML_mode = {'CNN_Train': False,
             'CNN_Pred' : True,
             'Sampler': True}
  
  # Global variables
  global num_pics
  num_pics = {'1Face': 0,
              '2Faces': 0,
              '3Faces': 0}
  global labels
  labels = {'1Face': 0,
            '2Faces': 1,
            '3Faces': 2}
  global num_data
  num_data = 0
  global splitsize # Fraction of data to build the training set
  splitsize = 0.7 
  global maxheight # Resize the pictures to a power of 2 for CNN (2^8 here)
  maxheight = 128
  global maxwidth
  maxwidth = 128
  global TreshEdge # Number of consecutive black pixels to define an edge
  TreshEdge = 2
  global TreshFace # Number of white pixels to define a face (or large edge)
  TreshFace = maxheight/16


  
  
  
  
  
  
  
  
