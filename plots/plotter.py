import numpy as np
import matplotlib.pyplot as plt

plotlist = ['Ref05-1.txt','Ref05-2.txt','Ref05-3.txt']
counter = 0
f, axarr = plt.subplots(3, 2,figsize=(14, 14))

# plot1
for item in plotlist:
  with open(item) as file:
    data = [[float(digit) for digit in line.split()] for line in file]
  array2d = np.asarray(data)

  axarr[counter,0].plot(array2d[:,0], array2d[:,1], 'b', array2d[:,0], array2d[:,3], 'r', alpha=0.75)
  axarr[counter,0].set_xlabel('Epochs')
  axarr[counter,0].set_ylabel('Loss')
  axarr[counter,0].legend(['Train','Validation'],loc=0,prop={'size': 10})

  axarr[counter,1].plot(array2d[:,0], array2d[:,2], 'b', array2d[:,0], array2d[:,4], 'r', alpha=0.75)
  axarr[counter,1].set_xlabel('Epochs')
  axarr[counter,1].set_ylabel('Accuracy')
  axarr[counter,1].legend(['Train','Validation'],loc=0,prop={'size': 10})
  
  if counter==0:
    axarr[counter,0].set_title('Runs 1 to 3 (top to bottom)')
    axarr[counter,1].set_title('Runs 1 to 3 (top to bottom)')
  counter += 1

plt.show()
