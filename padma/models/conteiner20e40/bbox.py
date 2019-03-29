import os
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from PIL import Image

# É preciso clonar o repositório models do tensorflow no raiz do padma
# para utilizá-lo https://github.com/IvanBrasilico/models.git
sys.path.append('./models/research')
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

SIZE = (600, 240)

PATH_TO_CKPT = os.path.join(os.path.dirname(
    __file__), 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(os.path.dirname(__file__), 'classes.pbtxt')
NUM_CLASSES = 2
TEST_IMAGE1 = os.path.join(os.path.dirname(__file__), '..', '..',
                           'tests', 'stamp1.jpg')
TEST_IMAGE2 = os.path.join(os.path.dirname(__file__), '..', '..',
                           'tests', 'stamp2.jpg')


def create_category_index():
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def create_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        # A linha abaixo faz a predição usar a CPU ao invés da GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = \
                        tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates
                #  to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                    real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                    real_num_detection, -1, -1])
                detection_masks_reframed = \
                    utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes,
                        image.shape[0],
                        image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.8), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name(
                'image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={
                                       image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays,
            #  so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = \
                [int(classe) for classe in output_dict['detection_classes'][0]]
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = \
                output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = \
                    output_dict['detection_masks'][0]
    return output_dict


def run_inference_for_batch(image, graph):
    with graph.as_default():
        # A linha abaixo faz a predição usar a CPU ao invés da GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = \
                        tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates
                #  to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                    real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                    real_num_detection, -1, -1])
                detection_masks_reframed = \
                    utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes,
                        image.shape[0],
                        image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.8), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name(
                'image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={
                                       image_tensor: image})

    return output_dict


class SSDMobileModel():
    """Object Detection Model trained to detect containers.
    """

    def __init__(self, threshold=0.8):
        """Args:
            threshold: only detection_boxes with more than threshold confidence
                will be returned
        """
        self._model = create_graph()
        self._labels_to_names = create_category_index()
        self._threshold = threshold
        (im_width, im_height) = SIZE
        self.input_shape = (im_height, im_width, 3)

    def predict(self, image):
        image_np = load_image_into_numpy_array(image)
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(
            image_np, self._model)
        result = []
        for ind, score in enumerate(output_dict['detection_scores']):
            if score > self._threshold:
                yfinal, xfinal, _ = image_np.shape
                bbox = [0., 0., 0., 0.]
                bbox[0] = int(output_dict['detection_boxes'][ind][0] * yfinal)
                bbox[2] = int(output_dict['detection_boxes'][ind][2] * yfinal)
                bbox[1] = int(output_dict['detection_boxes'][ind][1] * xfinal)
                bbox[3] = int(output_dict['detection_boxes'][ind][3] * xfinal)
                result.append({
                    'bbox': bbox,
                    'class': output_dict['detection_classes'][ind]
                })
        return result

    def prepara(self, image):
        image = image.resize(SIZE, Image.ANTIALIAS)
        return np.array(image.getdata()).reshape(*self.input_shape).astype(np.uint8)

    def predict_batch(self, batch, original_images):
        output_dict = run_inference_for_batch(batch, self._model)
        result = defaultdict(list)
        for ind, (scores, image) in enumerate(zip(output_dict['detection_scores'], original_images)):
            for ind2, score in enumerate(scores):
                if score > .8:
                    xfinal, yfinal = image.size
                    bbox = [0., 0., 0., 0.]
                    bbox[0] = int(output_dict['detection_boxes'][ind][ind2][0] * yfinal)
                    bbox[2] = int(output_dict['detection_boxes'][ind][ind2][2] * yfinal)
                    bbox[1] = int(output_dict['detection_boxes'][ind][ind2][1] * xfinal)
                    bbox[3] = int(output_dict['detection_boxes'][ind][ind2][3] * xfinal)
                    result[ind].append({
                        'bbox': bbox,
                        'class': int(output_dict['detection_classes'][ind][ind2])
                    })
        return result


if __name__ == '__main__':
    model = SSDMobileModel()
    images = []
    for image_path in [TEST_IMAGE1, TEST_IMAGE2]:
        image = Image.open(image_path)
        images.append(image)
    s0 = time.time()
    for r in range(3):
        for image in images:
            s = time.time()
            print(model.predict(image))
            print(time.time() - s)
    print('Tempo total', time.time() - s0)
