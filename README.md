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
![architecture](https://github.com/GuiguiJayJay/Zeolith/tree/master/plots/Architecture-Ref05.png)
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
#### Features extraction

### About the result
