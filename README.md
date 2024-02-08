## Facial Expression Recognition using CNN 
### Overview:
The project focuses on recognizing emotions from facial expressions in real-time scenarios.

To tackle this problem of emotion recognition, VGG16, and a customized 3-layer CNN are explored.

Augmentation techniques are employed to enhance the performance of the models. These techniques introduce variations in the images, improving the models' ability to generalize to different facial expressions.

The main objective is to identify the best performing model among the explored architectures based on accuracy .

The optimal model is further utilized to detect emotions from live video streams using OpenCV & webcam. This application demonstrates the practical use of the developed model in real-world scenarios

### Dataset Description:
 ➢ The project utilizes the FER 2013 dataset, which consists of grayscale images of faces. Each 
image is sized at 48x48 pixels.

 ➢ The training set consists of 28,709 examples, while the public test set contains 3,589 examples.
 
 ➢ The dataset includes seven emotion categories for classification: Angry (0), Disgust (1), Fear 
(2), Happy (3), Sad (4), Surprise (5), and Neutral (6)
![image](https://github.com/Shahriarmsakib/Facial-Expression-Recognition-using-CNN/assets/114893131/49920c0c-6f5c-4f97-a5d3-4fb6a38b692c)

### Work-Flow:
![image](https://github.com/Shahriarmsakib/Facial-Expression-Recognition-using-CNN/assets/114893131/48c86801-76cd-41ae-9467-5f86d11f782b)

### Expression Recognition in Real-Time:
 ➢ As  VGG16 model with data augmentation performs best, we utilized this model for real
time prediction of facial expressions.

 ➢ First we captured live video stream from a webcam using OpenCV and Python.
 
 ➢ Detected faces in each frame, crop the face and resized them to 48x48 grayscale 
images, matching the model's input requirements.

 ➢ Model predicted a number representing an expression, which was mapped to the 
corresponding expression and displayed alongside a related emoji

### Expression Recognition in Real-Time:
![image](https://github.com/Shahriarmsakib/Facial-Expression-Recognition-using-CNN/assets/114893131/cb5db028-e301-413f-a7ab-c468035a5b0b)
![image](https://github.com/Shahriarmsakib/Facial-Expression-Recognition-using-CNN/assets/114893131/58d97007-2a70-4c06-8669-5ec7631cdd0b)



