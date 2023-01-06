import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import remotezip as rz

import tensorflow as tf

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 5):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    print(self.path)
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
    
  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      #print(path)
      video_frames = frames_from_video_file(path, self.n_frames) 
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label

#fg = FrameGenerator(pathlib.Path("Splitted_data/train"), 10, training=False)

# for i in range(10):
#   frames, label = next(fg())
#   print(f"Shape: {frames.shape}")
#   print(f"Label: {label}")
#   for i in frames:
#     print(np.isnan(np.min(i)))
#     print(np.sum(i))

#print(frames[0])


# # Create the training set
# output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
#                     tf.TensorSpec(shape = (), dtype = tf.int16))
# train_ds = tf.data.Dataset.from_generator(FrameGenerator(pathlib.Path("Splitted_data/train"), 10, training=True),
#                                           output_signature = output_signature)

# for frames, labels in train_ds.take(10):
#   print(labels)

# # Create the validation set
# val_ds = tf.data.Dataset.from_generator(FrameGenerator(pathlib.Path("Splitted_data/val"), 10),
#                                         output_signature = output_signature)

# # Print the shapes of the data
# train_frames, train_labels = next(iter(train_ds))
# print(f'Shape of training set of frames: {train_frames.shape}')
# print(f'Shape of training labels: {train_labels.shape}')

# val_frames, val_labels = next(iter(val_ds))
# print(f'Shape of validation set of frames: {val_frames.shape}')
# print(f'Shape of validation labels: {val_labels.shape}')

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
# val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

# train_ds = train_ds.batch(2)
# val_ds = val_ds.batch(2)

# train_frames, train_labels = next(iter(train_ds))
# print(f'Shape of training set of frames: {train_frames.shape}')
# print(f'Shape of training labels: {train_labels.shape}')

# val_frames, val_labels = next(iter(val_ds))
# print(f'Shape of validation set of frames: {val_frames.shape}')
# print(f'Shape of validation labels: {val_labels.shape}')

# net = tf.keras.applications.EfficientNetB0(include_top = False)
# net.trainable = False

# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(scale=255),
#     tf.keras.layers.TimeDistributed(net),
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.GlobalAveragePooling3D()
# ])

# model.compile(optimizer = 'adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#               metrics=['accuracy'])

# model.fit(train_ds, 
#           epochs = 10,
#           validation_data = val_ds,
#           callbacks = tf.keras.callbacks.EarlyStopping(patience = 10, monitor = 'val_loss'))

# print(1)
#UCF101_subset/train/ApplyEyeMakeUp/v_ApplyEyeMakeup_g01_c03.avi
# sample_video = frames_from_video_file(pathlib.Path("Splitted_data/train/backhand/p2_backhand_s1.avi"), n_frames = 10)
# # #print(sample_video.shape)

# def to_gif(images):
#   converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
#   imageio.mimsave('./animation1.gif', converted_images, fps=10)
#   return embed.embed_file('./animation1.gif')

# to_gif(sample_video)
# fg = FrameGenerator(pathlib.Path("Splitted_data/train"), 10, training=True)

# print(list(fg()))

# frames, label = next(fg())
# to_gif(frames)


# print(f"Shape: {frames.shape}")
# print(f"Label: {label}")
# print(1)