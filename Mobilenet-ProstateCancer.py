import os
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU
import itertools
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix, classification_report
import scipy
import keras.backend as K

batchsize = 256
learningrate = 0.0001
optimizador = "SGD"

#Poner rutas donde se encuentra la carpeta de entrenamiento, validacion y testeo
train_path = "/Users/acuna/Desktop/Tesis/BBDD CON AUGMENTATION INDIVIDUAL/train"
valid_path = "/Users/acuna/Desktop/Tesis/BBDD CON AUGMENTATION INDIVIDUAL/valid"
test_path = "/Users/acuna/Desktop/Tesis/BBDD CON AUGMENTATION INDIVIDUAL/test"

#pre-procesando la base de datos
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), batch_size=batchsize)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), batch_size=batchsize)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), batch_size=batchsize, shuffle=False)


#Armando el modelo
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
optimizers = tf.keras.optimizers.legacy.SGD(learning_rate=learningrate, momentum=0.9)


x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=optimizers, loss="binary_crossentropy", metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)

train_loss_values = []
train_accuracy_values = []
val_loss_values = []
val_accuracy_values = []

print("---------------------------------------------------------------------------------------------")
print(" ")
history = model.fit(train_batches, validation_data=valid_batches, epochs=100, callbacks=[early_stopping])
train_loss_values.extend(history.history['loss'])
train_accuracy_values.extend(history.history['accuracy'])
val_loss_values.extend(history.history['val_loss'])
val_accuracy_values.extend(history.history['val_accuracy'])
print(" ")
print("---------------------------------------------------------------------------------------------")
model_save_path = "/Users/acuna/Desktop/Tesis/Codigos/modelocontransferlearnign.h5"
model.save(model_save_path)

print("---------------------------------------------------------------------------------------------")
print(" ")
test_loss, test_accuracy = model.evaluate(test_batches)
print("Test Accuracy:", test_accuracy)
print(" ")
print("---------------------------------------------------------------------------------------------")
# Make predictions on the validation data
print("---------------------------------------------------------------------------------------------")
print(" ")
y_pred_prob = model.predict(test_batches)
print(y_pred_prob)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions (assuming binary classification)
print(" ")
print(" ")
print("---------------------------------------------------------------------------------------------")
# Get true labels for validation data
y_true = test_batches.classes
print("---------------------------------------------------------------------------------------------")
print(" ")
print(" ")
# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print(" ")
print(" ")
print("---------------------------------------------------------------------------------------------")
# Extract TP, TN, FP, FN from the confusion matrix
TP = cm[1, 1]  # True Positive
TN = cm[0, 0]  # True Negative
FP = cm[0, 1]  # False Positive
FN = cm[1, 0]  # False Negative
print("---------------------------------------------------------------------------------------------")
print(" ")
print(" ")
# Print the results
print("True Positive (TP):", TP)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)
print(" ")
print(" ")
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print(" ")
print(" ")
presicion = TP/(TP + FP)
recall = TP / (TP + FN)
f1 = (2*presicion*recall)/(presicion + recall)
print("Presicion: ", presicion)
print("Recall: ", recall)
print("F1: ", f1)
print(" ")
print(" ")
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print(" ")
print(" ")
# Calculate classification report (includes precision, recall, F1-score, and accuracy)
report = classification_report(y_true, y_pred, target_names=valid_batches.class_indices)
print("Classification Report:")
print(report)
print(" ")
print(" ")
print("---------------------------------------------------------------------------------------------")
# Plot accuracy vs val_accuracy
#plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy VS Epoch')
plt.show()

# Plot loss vs val_loss
#plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Validation Loss VS Epoch')
plt.show()


np.savetxt("train_loss_values2.txt", train_loss_values)
np.savetxt("train_accuracy_values2.txt", train_accuracy_values)
np.savetxt("val_loss_values2.txt", val_loss_values)
np.savetxt("val_accuracy_values2.txt", val_accuracy_values)









