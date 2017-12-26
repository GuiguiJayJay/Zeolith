# Zeolith
The aim of this project was to perform object recognition on different kind of cubes, 
in order to find the number of visible faces. The data are confidential and are not provided,
that's why this time there won't be any Jupyter Notebook. I'll describe the feature engineering
in a dedicated section later.

Each picture consist of a cube, with 1 up to 3 faces visible (so the labels range from 1 to 3). 
The pictures can be very noisy and the faces of cubes can present minor to major defects making 
the identification not so straightforward. Finally, we only had about 280 pictures to solve this problem,
which is actually the main difficultyt of the task.

For this computer vision problem, I decided to go for a CNN to start with, then stacked a small DNN
on the top of it to process home-made features, both of them implemented via TensorFlow. I will go back on this later. 
At the end, this script reached a mean test accuracy of 67.8%. It is not great, but there is room for improvements.

## Architecture
I usual, I performed a lot of architecture testing, but there are still many possibilities to be checked.
The current architecture is the following:
[architecture](https://github.com/GuiguiJayJay/Zeolith/tree/master/plots/Architecture-Ref05.png)
where F stands for filter size, and S for the stride.

### Activation functions
I tried a big bunch of activation functions combination, but couldn't get any result as good the one we get with
ReLU. That being, given the number of testing examples (about 80), a few percent for accuracy doesn't means a lot.
I finally kept the ReLU for its discriminative power, as it push to 0 anything negative. This property can induce 
sparsity in the hidden units, allowing them to be more distinct. On the top of that, it doesn't face vanishing 
gradients problem given the form of its derivative (see [this paper](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)).

### Dimensionality of the NN and regularization
As stated Alex Karpathy for the CNN part, it's better to go for deeper CNN rather than wider CNN in case you need to increase
the network size. I could confirm this! Using wider CNN made the final testing accuracy worse, and drastically increased
training time. For the DNN part, it was pretty much the same. I started with 4000 units as the first DNN layer, and
tested 1000 and 500 units. The 2000 units gave me the best results, but there is much more work required here as I
didn't tried many layers configuration for this part. I sticked at 4-5 hidden layers made of 50 to 500 units and a droput 
of 0.1 to 0.9. A dropout of 0.2 gave me the best result, but the network still overfits a lot after some epochs
as you can see [here](https://github.com/GuiguiJayJay/Zeolith/tree/master/plots/Plots-Ref05.png). Going for
a deeper network didn't improve much within the previously mentionned range. A thiner network really degraded the 
accuracy though. A wider network didn't improve anything, but drastically increased training time, of course. I still 
have to test L2 regularization on weights.

### Mini-batch size
Using a batchsize above 1 really degrades the performances. We anyway don't have enough training examples to make this
technique really relevant. Using a batchsize above 1 helps generalizing behaviour observed in data by updating weights
only after after seeing bunch of examples. It speeds up the training and make it quite insensitive to local noises, 
but also avoid detailed exploration of parameters space, which in turn is usefull to try finding the global minimum
(see [this paper](https://arxiv.org/pdf/1206.5533.pdf)). 
In the current case, my training and testing scores were both quite poor but very close, and with a quite low loss, 
which means the network failed to reproduce most examples, but not by a big margin. Using a pure stochastic gradient 
descent yielded best results most likely because the deeper parameters space exploration allowed for a more precise
reconstruction of pictures specificity rather than an average trend. I thus decided to go for batches of size 1.

### Learning parameters
I used a learning rate of 0.0001, and a decay rate of 0.95 conditionned by the ratio of losses from epoch i+1 to epoch i. 
If the loss was progressing by less than 5% of the previous loss, then the decay rate was applied. This allowed for a 
slower but safer convergence. That being said, as we will show later, the loss still varies a lot depending on the run.
Using a bigger decay rate (or no decay rate at all) just made things worse. I ran the training several times and only kept 
averages of results for this reason. Every run was made of 50 epochs, even though in practice we observe overfit after 
20 epoch at most. I also used such big number of epochs for safety, and for the sake of curiosity (posterior analysis).
I plan to implement an interrupt switch to stop learning when we start to overfit, which should yield a little bit over
70% test accuracy according to preliminary calculations. I left neurons with their default initialization from TensorFlow.
Finally, I kept a random set of about 200 pictures as training set, and the remaining 80 as the test set.

### Loss
Nothing new or particular here. I used the cross-entropy error function for this 3-class problem, with a softmax 
function to extract the probability distribution across the classes.

### Feature engineering
I usually tend to separate features engineering into two categories: features reduction and features extraction.
In the first one, I try to compress the data, cutting outliers or reducing dimensionnality of the problem. In the 
second one, I try to build new features from the data to feed my neural networks with more explicit and/or relevant
features, as those kind of networks are extremely sensitive and need to as assisted as possible to get results.

#### Features reduction
The picutres provided looked like grayscaled pictures but still had 3 channels. The first obvious step was to remove
those additionnal channels. The second step was to rewrite the picture in a clearer way. So I answered this question:
how do you define a cube? By its edges or its faces. We basically only need a binary picture, were pixels are 0 for faces
and 1 for edges. At least in a perfect world. I therefore used OpenCV library to apply various filters in order to 
extract the edges:
- Laplacian transformation to extract all variations on the pictures (edges are typically the sharpest variations,
but some noises still sneak through)
- Bilateral filter to smooth noises to some extand without losing our previously found edges
- Adaptive Threshold to binarize the picture into edges, and flat areas

I then resizes all the pictures to 128x128 pixels. When using a CNN, it is a good practice to resize the pictures to 
high multiple of 2 to make pooling and most generally strides work more smoothly.

The architecture described above yields a mean test accuracy of 47.8% without this feature engineering, 
and about 60% with it. That's a good change (and it runs faster also).

#### Features extraction
This process is still at early stages, so let's say it is still a prototype. As I mentionned before, I think a good way 
to go is to start by thinking that a cube on a picture is defined by its edges and visible faces. In order to assist
even further the model in its learning, I decided to plug some explicit features on it after the CNN part, thus the DNN 
stacked over it. So basically, features created in this paragraph are concatenated with the logits in the second hidden 
layer of the DNN ("Dense2" in [architecture](https://github.com/GuiguiJayJay/Zeolith/tree/master/plots/Architecture-Ref05.png)) 
and are processed by subsequent layers.

The idea here is to look at 2 vertical and 2 horizontal slices of a given picture at equal distance from the center.
Then a simple edge detector algorithm (2 consecutive bins above 1 at the moment) returns the length of the 4 biggest
flat areas for each slices (flat below some threshold are discarded), yielding 16 features. Those features are then plugged to the DNN part of the model. Of course
all those settings need to be adjusted, it is just a prototype. Why only 2 slices? What if we use 3 consecutive black bins
as edge definition? What if we use a mean of longest flat areas across more than 2 slices? And so on.

Anyway, using those 16 new features help us to improve our mean test accuracy to 67.8% from about 60%. That's a nice result
especially given the very small amount of new features introduced. This option thus needs to be investigated much deeper.

### About the result
As you can see [here](https://github.com/GuiguiJayJay/Zeolith/tree/master/plots/Plots-Ref05.png), results are not so 
good yet. The model clearly overfits after 5-15 epochs. It is also very unstable as it produces quite different results
from one run to another. We are very sensitive to the random training set itself which is quite normal, there are limits 
to what one can train with about 200 pictures. Having more pictures would help stabilize the results from one run to 
another and data augmentation should be performed to start with. One can also see that the loss varies quite a lot
from one run to another. I tried to fix this unstability by adjusting learning and decay rates, but could only reduce
the effect up to now.

As I mentionned several times before, there is much to dig here. First, digging deeper the current features extraction, 
and searching for new features to be extracted. I also didn't tried L2 regularization on weights nor implemented a 
stop learning switch to prevent overfitting, meaning an increasing of test loss.

## Requirements
I created this script using:
- Python 3.6.2
- Scikit-Learn 0.19.0
- Pandas 0.20.3
- TensorFLow 1.2.1
- OpenCV 3.3.1

## Usage
First, you will need of course data. In the current layout, label is derived from the directory name in which you can 
find the pictures. Namely, you will need in the `data` directory the `1Face`, `2Faces` and `3Faces` subdirectories.
If you want to use weights from the disk and do predictions, you will need to additionnaly create a `test` subdirectory
and put your test pictures in.

Once this is done, simply execute the `main.py` script to run the specified tasks. You will need for this to set some 
options and or parameters, which can be found in 3 main places:
- `main.py`: change the training parameters at the `run_cnn` function call (batch size, intial learning rate, decay rate, 
number of epochs and dropout rate).
- `zeolib/models.py`: change the neural network architecture here.
- `zeolib/settings.py`: set the global parameters here. The `ML_mode` dictionnary typically needs to be looked at. The 2 
first options are to enable or not the training (if not, loads weights) and the prediction mode, and the last one, `Sampler`,
if features extraction is enabled or not. You can also change the split size between training and test set from the full 
dataset, the number of pixels needed to define an edge or a face for the `Sampler` mode and the picture resizing width/height.

## The files
- `main.py`: the main script.
- `preproc.py`: the script containing the functions to preprocess the data (apply filters, extract features, resize pictures).
- `settings.py`: the script containing the main options (such as perform the training or load weights, run in prediction mode,
the number of black pixels for edge detector, split size etc...)
This script also caintains the global variables declaration.
- `utils.py`: a script containing various utilities functions
- `models.py`: script containing the models used. All the tensorflow parts of the code can be found therein.






