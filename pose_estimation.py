# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def format_frames(frame,output_size):
    
    frame = tf.image.resize_with_pad(frame, *output_size)
    frame = tf.image.convert_image_dtype(frame, tf.int32)
    return frame

def video_to_skeleton(model,video_path,n_frames,output_size,frame_step):
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  
    start=50

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    cv2.imwrite('frame.jpg', frame)
    image = format_frames(frame, output_size)
    image = tf.convert_to_tensor(np.array([image]))
    print(image.shape)
    outputs = movenet(image)
    print(outputs)
    # for _ in range(n_frames - 1):
    #     for _ in range(frame_step):
    #         ret, frame = src.read()
    #     if ret:
    #         frame = format_frames(frame, output_size)
    #         result.append(frame)
    #     else:
    #         result.append(np.zeros_like(result[0]))
    src.release()

image_path = 'VIDEO_RGB_SUBSET/backhand/p3_backhand_s1.avi'
# Resize and pad the image to keep the aspect ratio and fit the expected size.
# image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("movenet_singlepose_lightning_4/")
movenet = model.signatures['serving_default']

video_to_skeleton(movenet,image_path,2,(192,192),1)
# # Run model inference.

# # Output is a [1, 1, 17, 3] tensor.
# keypoints = outputs['output_0']
# print(keypoints)