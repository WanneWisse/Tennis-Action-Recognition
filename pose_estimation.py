# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub

def format_frames(frame,output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame
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

image_path = 'nielsj.jpg'
image = tf.expand_dims(image, axis=0)
print(image.shape)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
# image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("movenet_singlepose_lightning_4/")
movenet = model.signatures['serving_default']

# Run model inference.
outputs = movenet(image)
# Output is a [1, 1, 17, 3] tensor.
keypoints = outputs['output_0']
print(keypoints)