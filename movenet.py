import tqdm
import random
import pathlib
import itertools
import collections
import os

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from FrameG import FrameGenerator

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model



def plot_results(history,model,num_frames,test_ds,modelname):
  #scores for table
  # print("recall: " + max(history.history['recall']))
  # print("precision: " + max(history.history['precision']))
 # print(history.history)
  print("accuracy: " + str(max(history.history['accuracy'])))
  #print("f1_score: " + max(history.history['f1_score']))

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
  plt.ylim([0,1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')

  plt.savefig("plot_"+modelname+".png", bbox_inches = "tight")  
  plt.clf()
  plt.close()
  #plt.show()

  #scores in confusionmatrix

  fg = FrameGenerator(pathlib.Path("Splitted_data/train/"), num_frames, training = True)
  label_names = list(fg.class_ids_for_name.keys())

  actual, predicted = get_actual_predicted_labels(model, test_ds)
  plot_confusion_matrix(actual, predicted, label_names, 'test',modelname)


  
def get_actual_predicted_labels(model, dataset):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type,modelname):
  
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize':(12, 12)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
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

def train_model(num_epochs,learning_r, batch_size, num_frames,num_layers_to_retrain):

  output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                      tf.TensorSpec(shape = (), dtype = tf.int16))

  train_ds = tf.data.Dataset.from_generator(FrameGenerator(pathlib.Path("Splitted_data/train/"), num_frames, training = True),
                                            output_signature = output_signature)
  train_ds = train_ds.batch(batch_size)

  test_ds = tf.data.Dataset.from_generator(FrameGenerator(pathlib.Path("Splitted_data/test/"), num_frames),
                                          output_signature = output_signature)
  test_ds = test_ds.batch(batch_size)

  for frames, labels in train_ds.take(10):
    print(labels)


  model_id = 'a0'
  resolution = 224

  tf.keras.backend.clear_session()

  backbone = movinet.Movinet(model_id=model_id)
  backbone.trainable = True

  # Freeze all the layers before the `fine_tune_at` layer
  #print( backbone.layers[:-num_layers_to_retrain])
  #print(backbone.layers)
  for layer in backbone.layers[:-num_layers_to_retrain]:
    layer.trainable = False

  for layer in backbone.layers:
    print(layer)
    print(layer.trainable)

  # Set num_classes=600 to load the pre-trained weights from the original model
  model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
  model.build([None, None, None, None, 3])

  checkpoint_dir = f'movinet_{model_id}_base'
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()

  model = build_classifier(batch_size, num_frames, resolution, backbone, 4)

  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  optimizer = tf.keras.optimizers.Adam(learning_rate = learning_r)



  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
  for layer in model.layers:
    print(layer)
  # print(model.layers[-3].get_weights())
  # print(model.layers[-2].get_weights())
  # print(model.layers[-1].get_weights())
  history = model.fit(train_ds,
                      validation_data=test_ds,
                      epochs=num_epochs,
                      validation_freq=1,
                      verbose=1,  callbacks=[callback])
  modelname = f'saved_model/retrained_model_{num_epochs}_{learning_r}_{num_frames}_{batch_size}_{num_layers_to_retrain}'
  plot_results(history,model,num_frames,test_ds,modelname)
  model.save(modelname)

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

learning_rates = [0.0001]
num_of_layers_to_retrain = [6]


for learning_rate in learning_rates:
  for layer in num_of_layers_to_retrain:
    train_model(num_epochs=100,learning_r=learning_rate,num_frames=16,batch_size=8,num_layers_to_retrain=layer)

