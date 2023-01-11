import os
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(history,modelname):
    print(history.history)

    #scores training vs val plot
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    #plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig("plot_"+modelname+".png", bbox_inches = "tight")  
    plt.clf()
    plt.close()

  
def get_actual_predicted_labels(model, xtest, y):  
  predicted = model.predict(xtest)
  
  actual = tf.stack(y, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels,modelname):
    print(actual)
    print(predicted)
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize':(12, 12)})
    sns.set(font_scale=1.4)
    #ax.set_title('Confusion matrix of action recognition for ' + ds_type)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig("conf_"+modelname+".png",  bbox_inches = "tight")
    plt.clf()
    plt.close()
    #plt.show()


CLASSES = ["backhand","smash","forehandopen","backhand2hands","backhandslice","forehandflat","forslice","serviceflat","servicekick","serviceslice","volley","volleybackhand"]
AMOUNT_FRAMES_TO_TAKE = 50
DIRECTORY = "normal_oniFiles"

samples_all_class = []
labels = []
for outer_d in os.listdir(DIRECTORY):
    #amateur/pro
    if outer_d == "ONI_EXPERTS":
        outer_d = os.path.join(DIRECTORY, outer_d)
        for d in os.listdir(outer_d):
            #if d in CLASSES:
            class_d = os.path.join(outer_d, d)
            for f in os.listdir(class_d):
                f = os.path.join(class_d, f)
                bp_x_vector = []
                bp_y_vector = []
                bp_z_vector = []
                #all frames for video
                vectors_for_all_frames = []
                file = open(f,"r")
                for line in file:
                    #print(line)
                    if "FRAME: 0" in line:
                        pass
                    elif "FRAME:" in line or line.strip() == '':
                        #we know have 1 frame
                        if sum(bp_x_vector)!= 0:
                            vectors_for_all_frames.append((bp_x_vector,bp_y_vector,bp_z_vector))
                        bp_x_vector = []
                        bp_y_vector = []
                        bp_z_vector = []
                    else:
                        bp_x,bp_y,bp_z = line.split()
                        bp_x_vector.append(float(bp_x))
                        bp_y_vector.append(float(bp_y))
                        bp_z_vector.append(float(bp_z))


                file.close()
                
                original_frames_to_take = []
                flipped_frames_to_take = []
                amount_frames = len(vectors_for_all_frames)
                for i in np.linspace(0, amount_frames-1, num=AMOUNT_FRAMES_TO_TAKE, dtype=int):
                    #horizontal flip and save sample
                    x,y,z = vectors_for_all_frames[i]
                    middle_x = (max(x)+min(x))/2
                    new_x_points = []
                    for point in x:
                        distance = middle_x-point
                        mirroredx = middle_x+distance
                        new_x_points.append(mirroredx)
                    original_frames_to_take.append(x+y+z)
                    flipped_frames_to_take.append(new_x_points+y+z)

                min_max_scaler = preprocessing.StandardScaler()
                original_frames_to_take = min_max_scaler.fit_transform(original_frames_to_take)
                original_frames_to_take = list(original_frames_to_take)
                for i in range(len(original_frames_to_take)):
                    original_frames_to_take[i] = list(original_frames_to_take[i])

                flipped_frames_to_take = min_max_scaler.fit_transform(flipped_frames_to_take)
                flipped_frames_to_take = list(flipped_frames_to_take)
                for i in range(len(flipped_frames_to_take)):
                    flipped_frames_to_take[i] = list(flipped_frames_to_take[i])

                samples_all_class.append(original_frames_to_take)
                samples_all_class.append(flipped_frames_to_take)
                labels.append(CLASSES.index(d))
                labels.append(CLASSES.index(d))

            #print(len(samples_all_class))

#print(len(samples_all_class))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(samples_all_class, labels, test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(type(y_test))

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(AMOUNT_FRAMES_TO_TAKE-1))

# Add a Dense layer with 10 units.
model.add(layers.Dense(len(CLASSES)))

#model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=["accuracy"],
)


history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), batch_size=20, epochs=250
)


actual,predicted = get_actual_predicted_labels(model,X_test,y_test)
print(predicted)
plot_confusion_matrix(actual,predicted,CLASSES,"lstm_test")
plot_results(history,"lstm_test")


            
            

