import numpy as np
import cv2
import time
import os
import urllib
import tarfile
import pickle
from config_v3_2 import *
import sys
# import tensorflow.contrib.tensorrt as trt
import numpy as np
import time
# from tf_trt_models.detection import *
from PIL import Image
import tensorflow as tf


NUM_CLASSES = 90
class Model(object):
  import tensorflow as tf
  def __init__(self, model_name):
    self.label_map_util = __import__('object_detection.utils')
    # Note, if you don't want to leak this, you'll want to turn Model into
    # a context manager. In practice, you probably don't have to worry
    # about it.
    if not os.path.isdir('data'):
      os,mkdir('data')
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    self.category_index = self.load_labels(PATH_TO_LABELS)
    self.model_path = MODEL_PATH
    self.model_name = model_name
    self.model_path_name = os.path.join(self.model_path, self.model_name)
    self.model_filename = self.model_name + '.tar.gz'
    print("Model.__init__ model_path_name", self.model_path_name)

    graph = None
    try:
      print("try model model_path_name", self.model_path_name)
      graph = self.load_model()
    except:
      print("Model not found: now downloading")
      self.download()
      graph = self.load_model()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    self.session = tf.Session(graph=graph, config=tf_config)

    ops = self.session.graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    self.tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
      ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
          self.tensor_dict[key] = self.session.graph.get_tensor_by_name(tensor_name)
    if 'detection_masks' in self.tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[1], image.shape[2])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
      # Follow the convention by adding back the batch dimension
      self.tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    self.image_tensor = self.session.graph.get_tensor_by_name('image_tensor:0')

  '''returns OUTPUT DICT or [(bbox, score)] for each detection of class target'''
  def predict(self, images, target=None):
    if len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    output_dict = self.session.run(self.tensor_dict,
                             feed_dict={self.image_tensor: images})
    if target is None:
      return output_dict
    ret = []
    for i in range(int(output_dict["num_detections"])):
      if str(self.category_index[output_dict['detection_classes'][0][i]]['name']) == target:
        ret.append((output_dict['detection_boxes'][0][i], output_dict['detection_scores'][0][i]))
    return ret

  def load_model(self):
    frozen_path = None
    if self.model_name[-3:] == '.pb':
      print("Check self.model_name[-3:] == .pb : Success")
      frozen_path = self.model_path_name
    else:
      frozen_path = os.path.join(self.model_path_name, 'frozen_inference_graph.pb')
    print("load model", frozen_path)
    graph_def = None
    with tf.gfile.GFile(frozen_path, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf_graph = tf.import_graph_def(graph_def, name='')
    return tf_graph 

  def download(self):
    model_filename = self.model_name + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    opener = urllib.request.URLopener()

    opener.retrieve(os.path.join(DOWNLOAD_BASE, model_filename), self.model_path_name + '.tar.gz')
    tar_file = tarfile.open(self.model_path_name + '.tar.gz')
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, self.model_path)

  def load_image_into_numpy_array(self, image, expand=True):
    if isinstance(image, str):
      image = Image.open(image)
    t = []
    t.append(time.time())
    (im_width, im_height) = image.size
    t.append(time.time())
    tmp = image.getdata()
    print(type(tmp))
    t.append(time.time())
    tmp = np.array(tmp)
    t.append(time.time())
    np_img = tmp.reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    t.append(time.time())
    print([t[i+1] - t[i] for i in range(len(t)-1)])
    return np.expand_dims(np_img, axis=0) if expand else np_img

  def load_image_into_numpy_array_cv(self, image, expand=True):
    t = []
    t.append(time.time())
    np_img = cv2.imread(image, 1)
    t.append(time.time())
    print(np_img.shape)
    np_img = np_img.astype(np.uint8)
    t.append(time.time())
    print([t[i+1] - t[i] for i in range(len(t)-1)])
    return np.expand_dims(np_img, axis=0) if expand else np_img

  def load_labels(self, path_to_labels=''):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def demo_from_memory():
  MODEL_NAME = "ssdlite_mobilenet_v2_coco_FP32_50_trt.pb"
  m = Model(MODEL_NAME)
  prev = time.time()
  timings = []
  only_pred = []
  for i in range(1,11):
    # img_orig = Image.open('data/imgs/{}.jpg'.format(i))
    img_name = 'data/imgs/{}.jpg'.format(i)
    # img = m.load_image_into_numpy_array(Image.open(img_name)) #pil_rect_smaller(img_name)
    img = m.load_image_into_numpy_array_cv(img_name)
    tmp_pred = time.time()
    out_dict = m.predict(img, 'person')
    timings.append(time.time() - prev)
    only_pred.append(time.time() - tmp_pred)
    prev = time.time()
    print(out_dict)
    exit()
  print("Timings: ")
  print(timings)
  print("Only TF prediction (without image preprocessing)")
  print(only_pred)
  print("Average prediction: {}, of which TensorFlow {}".format(np.mean(timings[1:]), np.mean(only_pred[1:])))

if __name__ == "__main__":
  t = []
  # MODEL_NAME = "ssdlite_mobilenet_v2_coco_FP32_50_trt.pb"
  MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
  try:
    c = cv2.VideoCapture(0)
    m = Model(MODEL_NAME)

    # IMAGE_PATH = './data/huskies.jpg'
    # image = Image.open(IMAGE_PATH)

    cv2.namedWindow("test")
    while True:
      t.append(time.time())
      ret, frame = c.read()
      if ret:
        bbox = m.predict(frame, 'person')
        if bbox:
          size = frame.shape
          y1, x1, y2, x2 = bbox[0][0]
          x1 = int(float(x1)*float(size[1]))
          x2 = int(float(x2)*float(size[1]))
          y1 = int(float(y1)*float(size[0]))
          y2 = int(float(y2)*float(size[0]))
          frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 3)
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256==27:
          break
  finally:
    print("AVERAGE delay: {} s".format(np.mean([t[i+1]-t[i] for i in range(3, len(t)-1)])))
