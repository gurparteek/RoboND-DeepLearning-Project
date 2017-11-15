# Follow Me Project Write-up by Gurparteek Bath

## Network Architecture Breakdown:

![alt text](./misc_images/architecture.png)

#### 1. Architecture Composition
The above network architecture is comprised of 3 encoders, 1 1x1 convolutional layer and then 3 decoders with skip connections.

#### 2. Encoders and Decoders
The encoders extract the features from the image and the decoders essentially *upscale* the output of the encoders such that the output is the same size as the original image. This results in the segmentation of each pixel in the input image.
A decoder is a transposed convolution which is essentially a reverse convolution as it *un-does* the previous convolutions by swapping the order of forward and backward passes.

#### 3. 1x1 Convolutional Layer and Fully Connected Layer
The 1x1 convolutional layer is the one that preserves the spatial information. This happens because we replace the fully connected layer by a 1x1 convolutional layer that ensures that the output tensor remains 4D instead of flattening it to 2D which results in the loss of spatial information as no information about the location of the pixels is preserved.
A fully connected layer where each node is *fully connected* to the nodes of the previous layers is only used to answer yes/no questions for identifying *if* the target is there, not where.

#### 4. Skip Connections
When encoders are employed, the effect of convolutions is that some information is still lost even if we upscale using decoders as it is the way that convolutions work by narrowing down and looking at different *smaller* features of the input image and losing a part of the *bigger picture*. Skip connections helps us retain that information by connecting the output of one layer to a non-adjacent layer which is then fed to the next layer. The output from different resolutions is combined and this helps the network make better segmentation decesions.

#### 5. Requirement for such Architecture
This is required as our purpose for the project is not only to answer a yes/no question to if the target is in the scene, we also want to *locate* the target in the scene. This requires the spatial information to be preserved through the entire network.

#### 6. Additional Benefit of such Architecture
Such an architecture has another advantage in that the size of the image input and output is the same and any size images can be fed into this architecture.

#### 7. Additional Techniques Employed
##### 7.1 Separable Convolutions
The encoders employ a technique called Separable convolutions or depthwise separable convolutions where convolutions are performed over each channel of the input layer, followed by a 1x1 convolution that takes the output from the prevuious channel and combines it into an output layer.
This results in a reduction of the number of parameters needed and increases the efficiency of the encoder network.
##### 7.2 Batch Normalization
With batch normalization, instead of normalizing only the input to the network, the input to each layer within the network is normalized and within the network, the output from one layer, becomes the input to another.
Batch normalization allow the networks to train faster and allow for higher learning rates.

## Hyperparameters:

#### 1. Learning Rate
Learning rate is the parameter which determines how fast our network can *learn* or make changes to the weights while it is being trained (minizing loss). In theory a larger learning rate would mean that the changes are incurred faster, or a greater change is observed in each step but it can be observed that the changes may also plateau out faster whereas a network with a smaller learning rate may start out small but keep changing and end up being better in minizing the loss.
I started out with a learning rate of 0.05 and eventually settled at 0.01 as that provided with better results.

#### 2. Batch Size
Batch size is the number of training images that are passed through the network in a single pass. The upper limit to choosing a batch size is limited by computing resources and should not be so small that the network has a difficulty distinguishing out the features.
I chose a batch size of 42 as I had 8300 training images and I chose the default 200 steps in an epoch, so to cover all the training resources, 8300/200 = ~42 was chosen.

#### 3. Number of Epochs
An epoch is one full forward and backward pass of the network over the data. Having multiple epochs increases the accuracy of the model without the requirement of additional data as the accuracy is increased with each epoch. There is however a saturation point beyond which there is no significant change in accuracy.
I started with number of epochs as 15 and increased them to 20 and then finally settled at 25 as the increase in accuracy was very small and going further would not bear much fruit.

#### 4. Steps per Epoch
This is the number of training data batches that will go through the network or that will be used for training in 1 epoch. If 1 batch is 42, then the network will train on 42 images in one step and if the number of steps per epoch is set to 200, then in 1 epoch, the network would have trained on 42 x 200 = 8400 images.
The default value of 200 was chosen for this and the batch size was adjusted accordingly.

#### 5. Validation Steps
This is similar to steps per epoch and is the number of batches of validation images that go through the network in 1 epoch.
For ~1200 validation images and 42 batch size, 30 validation steps were chosen to cover the data.

### Model Limitations (for use with other objects)
This model (using the available data) has been trained to follow only one person and it in its present state (using the same trained weights) can not be used to follow any other object. The network however, is capable of being trained to follow another person or object given the hyperparameters may need tuning to suit the needs but the most important task would be to acquire data and then to train the network to identify that person or object.

### Results and Future Enhancements
My score incresed from an initial score of 33% to 41.2% by tuning the hyperparameters such as decreasing the learning rate from 0.05 to 0.01, increasing the number of epochs from 15 to 25 and increasing the batch size from 32 to 42. A good imporvement was doubling the training data by flipping all the images horizontally. Doubling the number of filters in the 1x1 convolutional layer also bumped my score up a bit.
The most important enhancement that I can think of for my model would be to expand the training data and collect more useful data such as that of the hero from afar. Other than this, further tuning of hyperparameters would increase the accuracy of the model which I was limited to test due to the lack of computational resources.

### Submissions
1. A final score of 41.2% was achieved as can be seen in the model_training.ipynb notebook in './code'.
2. An HTML version of model_training.ipynb notebook has been provided in the main directory of the repo.
3. Weights file in the .h5 file format has been provided in './data/weights'.