import os
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
import time
from datetime import datetime

def plot_results(history,modelname):
    #print(history.history)

    #scores training vs val plot
    acc = []
    val_acc = []

    loss = []
    val_loss = []

    for h in history:
        acc.append(h.history['accuracy'])
        val_acc.append(h.history['val_accuracy'])

        loss.append(h.history['loss'])
        val_loss.append(h.history['val_loss'])

    acc = np.array(acc)
    acc = acc.mean(axis=0)
    print(acc)

    val_acc = np.array(val_acc)
    val_acc = val_acc.mean(axis=0)

    loss = np.array(loss)
    loss = loss.mean(axis=0)

    val_loss = np.array(val_loss)
    val_loss = val_loss.mean(axis=0)

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

    plt.savefig("lstm_models/plot_"+modelname+".png", bbox_inches = "tight")  
    plt.clf()
    plt.close()

  
def get_actual_predicted_labels(model, xtest, y):  
  predicted = model.predict(xtest)
  
  actual = tf.stack(y, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels,modelname):
    # Print the confusion matrix
    cm = metrics.confusion_matrix(actual, predicted)

    cm = cm.astype('float') / 5

    cm = cm.astype('int')

    # Print the precision and recall, among other metrics
    report = metrics.classification_report(actual, predicted, target_names=labels, digits=3, output_dict=True)
    f = open(f"scores/{modelname}.txt", "w")
    for label,scores in report.items():
        f.write(label+ ": ")
        if type(scores) == dict:
            for score_name,score_value in scores.items():
                f.write(score_name + ": " + str(score_value) + ", ")
        else:
            f.write(str(scores))
        f.write("\n")
    f.close() 
           
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
    plt.savefig("lstm_models/conf_"+modelname+".png",  bbox_inches = "tight")
    plt.clf()
    plt.close()


CLASSES = ["backhand","smash","forehandopen","backhand2hands","backhandslice","forehandflat","forslice","service","volley","volleybackhand"]
DIRECTORY = "normal_oniFiles-grouped-service"

def train_model(modelname, amount_frames_to_take, batch_size, learning_rate, model_rnn):
    samples_all_class = []
    labels = []
    for outer_d in os.listdir(DIRECTORY):
        #amateur/pro
        #if outer_d == "ONI_EXPERTS":
        outer_d = os.path.join(DIRECTORY, outer_d)
        for d in os.listdir(outer_d):
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
                for i in np.linspace(0, amount_frames-1, num=amount_frames_to_take, dtype=int):
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


    kf = KFold(n_splits=5, shuffle=True, random_state=3)
    
    #X = pd.DataFrame(samples_all_class)
    #print(X.head())
    X = np.array(samples_all_class)
    #y = pd.DataFrame(samples_all_class)
    y = np.array(labels)

    all_history = []
    all_actual = []
    all_predicted = []

    for train_idx, val_idx in kf.split(X, y):
        print("train indx: ", train_idx)
        print("Test indx: ", val_idx)
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[val_idx]
        y_test = y[val_idx]

        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        model = keras.Sequential()

        if model_rnn == "LSTM":
            model.add(layers.LSTM(amount_frames_to_take-1))
        else:
            model.add(layers.GRU(amount_frames_to_take-1))

        model.add(layers.Dense(len(CLASSES)))


        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=250
        )

        actual,predicted = get_actual_predicted_labels(model,X_test,y_test)

        all_history.append(history)
        all_actual = np.concatenate((all_actual, actual))
        all_predicted = np.concatenate((all_predicted, predicted))

    plot_confusion_matrix(all_actual,all_predicted,CLASSES,modelname)
    plot_results(all_history, modelname)


# hyperparamters: amount_frames_to_take, batch_size, learning_rate, model_rnn
model_rnn = ["LSTM","GRU"]
learning_rates = [0.001,0.0001]
batch_sizes = [8,16,32]
amounts_frames_to_take = [40,60]

for model in model_rnn:
    for learing_r in learning_rates:
        for batch_s in batch_sizes:
            for amount_f in amounts_frames_to_take:
                train_model(f"{model}_{learing_r}_{batch_s}_{amount_f}_crossVal",amount_f,batch_s,learing_r,model_rnn)