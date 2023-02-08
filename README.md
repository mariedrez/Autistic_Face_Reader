# Autistic Face Reader

## Background
During my stint as an early childhood educator, from time to time there were parents who approached the school about their concerns that their child may be presenting autistic qualities. Also known as Autism Spectrum Disorder, ASD is a lifelong neuro-developmental condition that is "characterised by persistent impairment in reciprocal social communication and social interaction, restricted, repetitive patterns of behaviour, interests, or activities"<a href="https://www.nuh.com.sg/Health-Information/Diseases-Conditions/Pages/Autism-(Children).aspx" target="_blank"><sup>1</sup></a>.
 Symptoms of ASD can surface during early childhood and impair activities of daily life<a href="https://thespectrum.org.au/autism-strategy/autism-strategy-activities-daily-living/" target="_blank"><sup>2</sup></a>. In Singapore, it was reported that an estimated 1 in 150 children are on the autism spectrum <a href="https://www.msf.gov.sg/policies/Disabilities-and-Special-Needs/Documents/Enabling%20Masterplan%203%20%28revised%2013%20Jan%202017%29.pdf" target="_blank"><sup>3</sup></a>, which outstrips the worldwide incidence of 1 in 100 children as reported by the World Health Organisation<a href="https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders" target="_blank"><sup>4</sup></a>.

While teachers can provide their input based on daily observational records, parents are usually advised to seek professional evaluation from either a child psychiatrist, a psychologist or a developmental paediatrician. In some cases, the teacher's anecdotal records on a child's behaviours in school can provide substantial evidence. However, in many preschools, there can be up to thirty students in a class rotating through various programmes, which may make it a tad challenging for a teacher to spot behavioural quirks in a child who is suspected to be on the spectrum. Additionally, autism masking is common among children with autism<a href="https://www.theautismservice.co.uk/news/what-is-autism-masking-or-camouflaging/" target="_blank"><sup>5</sup></a> whereby the child inevitably picks up and performs certain social behaviours, and actively conceal other types of behaviours in order to assimilate in social situations.

## Problem Statement
There are many obstacles that can hinder a preschool educator's ability to spot a child with signs and symptoms of autism. This includes the limitations that arise from a busy school setting and the ability of the child to mask the signs and symptoms for autism. An Autistic Face Reader can be built using Convolutional Neural Networks that takes in image data of a child and classifies the face as being "Autistic" or "Non-autistic". Inputting such a model into the daily attendance taking system can alert the teacher in-charge when a student's facial image data is consistently classified as "Autistic". This can lead them to take note of their behavioural tendencies and update their anecdotal records on the child daily. 

Ultimately, a positive "autistic" face reading would not replace a professional evaluation, but act as a useful screening step to alert preschool management to look into advising parents accordingly, such as ways to receive a formal diagnosis. Having a formal diagnosis will allow the child with autism (as well as their family) to understand their unique experiences and to receive proper support. Furthermore, the Autistic Face Reader helps to create a window of time before the formal diagnosis which the school can leverage to make necessary adjustments to accommodate their learning needs.  
<br>

`Success Metrics`
- <b> Precision </b>: calculated as the ratio of the number of "autistic" (Positive) samples correctly classified to the total number of samples classified as "autistic" (whether correctly or incorrectly). As such, precision measures the model's accuracy in classifying an image sample as "autistic".
    - Precision scores can be improved when:
        - The model classifies more "autistic" images correctly. (True Positive is maximised)
        - The model classifies less "autistic" images incorrectly. (False Positive is minimised)
- <b> Recall </b>: calculated as the ratio of the number of "autistic" (Positive) samples correctly classified to the total number of "autistic" samples. As such, recall measures the model's ability to detect Positive samples.
    - Recall scores can be improved when:
        - The model classifies more "autistic" images correctly. (True Positive is maximised)
        - The model classifies less "non_autistic" images incorrectly. (False Positive is minimised)
- For this project, we will consider it a success if the neural network can classify "autistic" image data correctly with precision and recall scores of at least 70%.

## Workflow

### Part 1: Data Import and Analysis
The image dataset which was obtained from Kaggle <a href="https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set" target="_blank"><sup>6</sup></a> has been adapted for use in this project. It comprises facial image data that belong to either of two classes: children with autism and children without autism. The folder titled 'consolidated' contains 1468 images per class, and 2936 images altogether.

- These are randomly sampled images from each class:
<img width="400" alt="Screenshot 2023-02-07 at 11 47 43 PM" src="https://user-images.githubusercontent.com/115085993/217459239-95954306-e930-4248-ae15-30f71d4754f8.png">

- Comparing the average facial image of a child without autism and a child with autism:
<img width="182" alt="Screenshot 2023-02-08 at 12 43 57 AM" src="https://user-images.githubusercontent.com/115085993/217459383-b9a24569-93ab-4b5d-b886-1ae3b79ce77f.png">

<img width="312" alt="Screenshot 2023-02-08 at 12 44 28 AM" src="https://user-images.githubusercontent.com/115085993/217459450-2c4f5f64-fa7d-4fe9-9d7b-58e8e4711cef.png">


- Comparing the variance in facial image data within each class
<img width="186" alt="Screenshot 2023-02-08 at 12 44 56 AM" src="https://user-images.githubusercontent.com/115085993/217459537-6651f88d-b962-4308-9575-c1cac36e5cd8.png">

<img width="258" alt="Screenshot 2023-02-08 at 12 51 31 AM" src="https://user-images.githubusercontent.com/115085993/217459600-32b22ce3-5ff3-4712-93b6-9194d36c6a7c.png">

### Part 2: Modelling
`Data Pre-processing and Augmentation` 

Pre-processing data include shuffling the training dataset and converting train_labels into a binary matrix using the method to_categorical(). The function returns a matrix of binary values (either ‘1’ for autistic or ‘0’ for non-autistic).

Image data augmentation is done with steps suchs as normalising pixel values between 0 and 1, randomly rotating images by 15 degrees, and flipping images 180 degrees horizontally. As the size of the training dataset is relatively small at less than 2000 images, data augmentation helps to increase its variability. In turn, this helps the model generalize better to new, unseen data and reduce the chances of model overfitting.


`Convolutional Neural Networks` <br>
A convolutional neural network (CNN) is a type of artificial neural network designed for processing structured arrays of data. Due to its ability to recognize patterns through deep learning, CNNs are widely used for image recognition and processing.

As a feed-forward neural network, an input image fed through a CNN is processed sequentially through several hidden layers, which are stacked together. In a typical CNN, the hidden layers are convolutional layers followed by activation layers, and of which some are followed by pooling layers. Each type of layer performs a specific operation on the image data.

- <b> Convolution layers</b>  push the input image through a set of convolutional filters, each of which activates certain features from the images.

- <b> Rectified linear unit (ReLU)</b> allows for faster and more effective training by mapping negative values to zero and maintaining positive values. This is sometimes referred to as activation, because only the activated features are carried forward into the next layer.

- <b> Pooling layers </b> simplify the output by performing nonlinear downsampling, reducing the number of parameters that the network needs to learn.

- <b> Dropout layers </b> are a regularisation technique. Randomly selected neurons are ignored during training, i.e. “dropped out” randomly, which helps prevent overfitting. 


`Callbacks` <br>
"A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training."

- <b> EarlyStopping </b> enables the termination of the training process when the model's performance on a validation dataset stops improving. This can help avoid overfitting, where the model stops generalising and begins to learn from statistical noise in the data.
- <b> ReduceLearningRateonPlateau </b> adjusts the learning rate when a plateau in model performance is detected. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

`Model 1: Custom CNN`

<img width="961" alt="Screenshot 2023-02-08 at 12 43 02 PM" src="https://user-images.githubusercontent.com/115085993/217459705-26bd40f7-c567-4fab-94d3-140ababccd90.png">

The training loss is decreasing and seems to continues to decrease at the end of the plot. This indicates that the model is underfitting and capable of further learning should the training process not been halted prematurely.

`Model 2: Transfer Learning (VGG 16)`
<img width="954" alt="Screenshot 2023-02-08 at 1 17 53 PM" src="https://user-images.githubusercontent.com/115085993/217459750-106cb0d7-686c-4b76-ba04-52f16a165149.png">

Here we find the validation loss is much better than the training loss. This reflects the validation dataset is easier for the model to predict than the training dataset. Furthermore, it is possible that the validation data is relatively small, but widely represented by the training dataset, thus the model performs extremely well on these few examples.

### Part 3: Model Score
`Confusion Matrix` 

<img width="291" alt="Screenshot 2023-02-08 at 1 21 24 PM" src="https://user-images.githubusercontent.com/115085993/217459799-20732e02-bd4e-4fc3-b38e-1fe40ba50fbd.png">

`Classification report`

<img width="358" alt="Screenshot 2023-02-08 at 1 22 06 PM" src="https://user-images.githubusercontent.com/115085993/217459821-09a412f6-70de-441f-b309-e2a25f77293b.png">

## Results
`Correctly classified images`

<img width="503" alt="Screenshot 2023-02-08 at 1 27 59 PM" src="https://user-images.githubusercontent.com/115085993/217459897-f4a16cfd-ac43-40a4-8012-fbeafe1e0b88.png">

`Wrongly classified images` 

<img width="524" alt="Screenshot 2023-02-08 at 1 28 31 PM" src="https://user-images.githubusercontent.com/115085993/217459937-a5ac0614-f167-4c27-9d0c-c7742a6a8a6b.png">

## Discussion
`Recommendations and Future Work`
- <b> Convert all image data to grayscale </b> <br>
Colour channels are a reflection of the image dataset's dimensionality. Grayscale images comprise just 1 channel, whereas most colour images (RGB) comprise 3 channels. Converting images to grayscale will reduce the complexity of the data and decrease time taken to train the model. 

- <b> Perform finetuning on transfer learning with VGG-16 </b> <br>
The goal of fine-tuning is to allow a segment of the pre-trained layers to retrain. Example of fine-tuning to do will be to compile the model with a lower learning rate.

`Conclusion` <br>
It has been a tremendous learning journey with Convolutional Neural Networks and Transfer Learning. I have learnt that it is possible to train a machine to read facial image data patterns and classify said image data with high precision and recall, albeit with immense effort and attention. In addition, data cleanliness is extremely crucial for the success of the neural network, thus, an increased dataset with higher reliability on input data for the positive class can improve precision and recall scores.
