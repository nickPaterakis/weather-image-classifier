import cv2
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from random import seed
from random import randint
import numpy as np
import itertools
import matplotlib.pyplot as plt
from LoadData import X_train, Y_train, X_test, Y_test

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# class 0 cloudy
# class 1 rain
# class 2 shine
# class 3 sunrise
classes = ["Cloudy", "Rain", "Shine", "Sunrise"]

# Load the model
loaded_model = keras.models.load_model('CNN_model.h5')

# Parse numbers as floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize data
X_train = X_train / 255
X_test = X_test / 255

y_test_predictions_vectorized = loaded_model.predict(X_test)
y_pred_test = np.argmax(y_test_predictions_vectorized, axis=1)

y_test_predictions_vectorized = loaded_model.predict(X_train)
y_pred_train = np.argmax(y_test_predictions_vectorized, axis=1)

acc_train = accuracy_score(Y_train, y_pred_train)
acc_test = accuracy_score(Y_test, y_pred_test)
pre_train = precision_score(Y_train, y_pred_train, average='macro')
pre_test = precision_score(Y_test, y_pred_test, average='macro')
rec_train = recall_score(Y_train, y_pred_train, average='macro')
rec_test = recall_score(Y_test, y_pred_test, average='macro')
f1_train = f1_score(Y_train, y_pred_train, average='macro')
f1_test = f1_score(Y_test, y_pred_test, average='macro')

# Generate generalization metrics
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

print('Accuracy scores:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
print('')

cm = confusion_matrix(Y_test, y_pred_test)
plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

seed(3)
class_to_demonstrate = 0
while (sum(y_pred_test == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(y_pred_test == class_to_demonstrate)

    # create new plot window
    plt.figure()

    image_num = len(tmp_idxs_to_use[0][:]) - 1

    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(X_test[tmp_idxs_to_use[0][randint(0, image_num)]], cv2.COLOR_BGR2RGB))
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(X_test[tmp_idxs_to_use[0][randint(0, image_num)]], cv2.COLOR_BGR2RGB))
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(X_test[tmp_idxs_to_use[0][randint(0, image_num)]], cv2.COLOR_BGR2RGB))
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(X_test[tmp_idxs_to_use[0][randint(0, image_num)]], cv2.COLOR_BGR2RGB))
    tmp_title = 'Images considered as ' + str(classes[class_to_demonstrate])
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1
