# CiTius-Summer-Research
Automatic detection of anomalies in arbitrary images

The aim of the research project is to perform a first analysis on whether anomaly detection models based on Convolutional Neural Networks can outperform the classical detection models based on Saliency maps (AWS and WMAP). Two CNN approaches are proposed:

- An inpainting CNN model which given an image with the center cropped the model outputs the same image with the crop painted based on its context.
- A Convolutional AutoEncoder which learns how to compress and uncompress an image to a reduced number of float values. Therefore, the reduced float values represent the structures and characteristics identified in the original image.

Data Augmentation techniques are used for the training process. For both the training and the evaluation of both models, a sliding window technique is used to drastically increment the number of training samples and to make anomaly detection possible. Given a set of windows from an image and given their outputs of the CNN model, a series of test and comparations are computed in order to decide, first, wheter an anomaly is present in the image and secondly, if an anomaly is detected, identify the windows that are more likely to have an anomaly.

A total of approximately 3.000 images are used for training and 750 for evaluation. All of the images come from a public dataset of tree bark images called "Bark-101". Different hyperparameter configurations are trained and tested on the anomaly detection problem. The file named "RESULTS AUC" provides different ROC curves altogether with their respective areas under the curve of the top performing trained models along with the classical Saliency-based models.
