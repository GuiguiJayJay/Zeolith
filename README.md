# Zeolith
The aim of this project was to perform object recognition on different kind of cubes, 
in order to find the number of visible faces. The data are confidential and are not provided.
Each picture consist of a cube, with 1 up to 3 faces visible (so the labels range from 1 to 3). 
The pictures can be very noisy and the faces of cubes can present minor to major defects making 
the identification not so straightforward. Finally, we only had about 280 pictures to solve this problem,
which is actually the main difficultyt of the task.

For this computer vision problem, I decided to go for a CNN to start with, then stacked a small DNN
on the top of it to process home-made features. I will go back on this later. At the end, this script reached
a mean accuracy of 67.8%. It is not great, but there is room for improvements.

## Architecture
I usual, I performed a lot of architecture testing, but there are still many possibilities to be checked.
The current architecture is the following:
![architecture](https://github.com/GuiguiJayJay/Zeolith/tree/master/plots/Architecture-Ref05.png)
