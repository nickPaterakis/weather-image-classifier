# Weather Image Classifier
## Overview
This project aims to classify weather conditions in images using deep learning techniques. The dataset used in this project is the Multi-class Weather Dataset which contains images of cloudy, rainy, sunny, and sunrise weather conditions.

The model used for classification is a VGG neural network implemented in Python using the Keras library. The model has achieved over 90% accuracy on the test set.

In addition to the classification score diagram, the project includes diagrams showing the validation accuracy and loss history as well as a confusion matrix. The repository also includes example images of the model's predictions.

## Performance Visualization & Evaluation Metrics

This section of the project is dedicated to presenting various diagrams that offer a comprehensive understanding of the model's performance.

### Classification Score Diagram

The Classification Score Diagram is an insightful visual representation that displays the accuracy, precision, F1 score, and recall metrics of our model. Each of these metrics gives us a different perspective on the model's performance, and together they provide a well-rounded evaluation of the model's prediction capabilities.

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118052373-47b80380-b38b-11eb-9989-f6529d11e2b6.png" alt="classification score diagram" width="600">
</p>

### Accuracy History and Validation Loss History

The Accuracy History and Validation Loss History diagrams give an overview of the model's learning progress throughout the training process. The Accuracy History diagram tracks the model's accuracy over the course of training epochs, while the Validation Loss History diagram shows how the model's loss changed during the same period.

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118052575-aaa99a80-b38b-11eb-890d-72e80b4bc3ab.png" alt="validation-accuracy" width="600">
</p>

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118052578-ab423100-b38b-11eb-886a-d19b6e871b26.png" alt="validation-loss" width="600">
</p>

### Confusion Matrix
The Confusion Matrix is a powerful tool that provides a summary of the prediction results on our test data. It shows the number of correct and incorrect predictions made by our model, classified by each type of weather condition. It is an excellent way to visualize the performance of our model, especially in terms of its ability to correctly predict each weather condition.

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118039791-63b2a980-b379-11eb-9de1-c0de2bebf85d.png" alt="confusion matrix" width="600">
</p>

## Visualizing the Predictions
Along with the task of calculating the evaluation metrics, this project includes an aspect of visualizing the model's predictions. Selected randomly, 4 images from each predicted class are showcased alongside their corresponding predicted weather condition. This provides us with a tangible output, enabling a more comprehensive understanding of our model's performance.

Take a moment to review these images and observe the accuracy of our model. It's exciting to see how accurately our model can predict the weather conditions from different images!

Remember that these are just a few samples, and the actual performance of the model can be understood in a detailed manner by looking at the computed evaluation metrics - accuracy, precision, recall, and F1-score.

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118039775-61504f80-b379-11eb-9410-86124c96c91a.png" alt="shine" width="600">
</p>

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118039871-7af19700-b379-11eb-9f51-063327ab53cc.png" alt="cloudy" width="600">
</p>

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118039752-5dbcc880-b379-11eb-879c-1d73fdd27184.png" alt="rain" width="600">
</p>

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118039794-64e3d680-b379-11eb-999f-b2e508a92edc.png" alt="sunrise" width="600">
</p>

## How to Run:
1. Ensure that you have all the necessary Python libraries installed. This includes cv2, keras, sklearn, numpy, matplotlib and tensorflow.

2. The dataset must be properly structured and the LoadData.py file must correctly load the data. The data should be split into training, testing, and validation sets.

3. Run createModel.py to train the CNN model. This will create a model file named 'CNN_model.h5'.


```bash
python createModel.py
```

4. Run CreateResults.py to make predictions on the test data and calculate the evaluation metrics.
```bash
python CreateResults.py
```

## Note
Please ensure that you have a compatible environment to run these scripts. This includes the correct versions of Python and the necessary libraries. Also, you will need to have sufficient computational resources, especially if your dataset is large. It is recommended to run these scripts on a machine with a good CPU, enough RAM, and a powerful GPU. You can also use cloud-based solutions for training your model.

