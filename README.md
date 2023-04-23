# weather-predictor
A VGG neural network that can predict if the weather in an image is cloudy, rainy, sunny or sunrise. It achieves to find correct more than 90% of the images in test set. I used Python, and Keras.

# Model Performance

The VGG neural network model achieved an accuracy of over 90% on the test set. The classification score diagram shows the accuracy, precision, F1 score, and recall metrics of the model:



<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118052373-47b80380-b38b-11eb-9989-f6529d11e2b6.png" alt="classification score diagram" width="600">
</p>

The validation accuracy history and validation loss history diagrams are shown below:



<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118052575-aaa99a80-b38b-11eb-890d-72e80b4bc3ab.png" alt="validation-accuracy" width="600">
</p>

<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118052578-ab423100-b38b-11eb-886a-d19b6e871b26.png" alt="validation-loss" width="600">
</p>

The confusion matrix diagram helps visualize the model's performance in predicting the different weather conditions:



<p align="center">
 <img src="https://user-images.githubusercontent.com/36018286/118039791-63b2a980-b379-11eb-9de1-c0de2bebf85d.png" alt="confusion matrix" width="600">
</p>

Examples of the model's predictions on new images are shown below:



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
