'''
Script for testing live inferences.
'''

import cv2
import numpy as np
import os
import math
import tensorflow as tf
import json
from model import build_EfficientPose
from utils import preprocess_image
from utils.visualization import draw_detections

CLASS_TO_NAME = {0 : "2-slot", 1 : "3-slot", 2 : "mushroombutton", 3 : "arrowbutton", 4 : "redbutton", 5 : "unknownbutton"}

def main():
    """
    Run EfficientPose in inference mode.
    
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()

    #input parameters
    phi = 0

    path_to_weights = "./weights/buttonpose.h5"
    path_to_color_images = "./datasets/ButtonPose/data/rgb/"
    start_number = 18000
    total = 20000

    class_to_name = CLASS_TO_NAME
    score_threshold = 0.5
    translation_scale_norm = 1000.0     # conversion factor from m to mm
    draw_bbox_2d = False
    draw_name = False

    capturedata = {}
    
    name_to_3d_bboxes = get_3d_bboxes()
    class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
    num_classes = len(class_to_name)

    #build model and load weights
    model, image_size = build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights)

    camera_matrix = get_camera_params()
    
    cv2.namedWindow('ScrewPose', cv2.WINDOW_NORMAL)

    #inferencing
    print("\nStarting inference...\n")


    for i in range(start_number, total):
        color_image = cv2.imread(os.path.join(path_to_color_images, str(i) + ".png"))
        
        # Preprocessing
        original_image = color_image.copy()
        input_list, scale = preprocess(color_image, image_size, camera_matrix, translation_scale_norm)
        
        # Pose inference with EfficientPose
        boxes, scores, labels, rotations, translations = model.predict_on_batch(input_list)
        
        # Postprocessing
        boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold)

        capture = parse_data(labels, rotations, translations, class_to_name)
        capturedata[i] = capture

        # Plotting detections and displaying the image
        draw_detections(original_image,
                        boxes,
                        scores,
                        labels,
                        rotations,
                        translations,
                        class_to_bbox_3D = class_to_3d_bboxes,
                        camera_matrix = camera_matrix,
                        label_to_name = class_to_name,
                        draw_bbox_2d = draw_bbox_2d,
                        draw_name = draw_name)
		
		
        cv2.imshow("ScrewPose", original_image)
		
		# Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break
    
    with open("capturedata.json", 'w') as jsonfile:
        json.dump(capturedata, jsonfile)

def allow_gpu_growth_memory():
    """image_size
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)

def get_camera_params():
    """
    Gets camera parameters.
    Args:
        print: bool, if true prints the calibration parameters when executed.
    Returns:
        mat: 3x3 camera intrinsic matrix of the type:
            |fx  0   cx|
            |0   fy  cy|
            |0   0    1|
    """
    return np.array([[640.0, 0.,  640.0], [0., 640.0, 360.0], [0.0, 0., 1.]], dtype=np.float32)

def get_3d_bboxes():
    """
    Returns:
        name_to_3d_bboxes: Dictionary with the Linemod and Occlusion 3D model names as keys and the cuboids as values

    """
    name_to_model_info = {"2-slot": {"diameter": 137.1, "min_x": -34.0 , "min_y": -52.7935, "min_z": -27.5, "size_x": 68.0, "size_y": 105.5870, "size_z": 55.0},
                          "3-slot": {"diameter": 161.36, "min_x": -34.0 , "min_y": -67.801, "min_z": -27.5, "size_x": 68.0, "size_y": 135.602, "size_z": 55.0},
                          "mushroombutton": {"diameter": 59.698, "min_x": -24.95, "min_y": -20.0, "min_z": -20.0, "size_x": 51.9, "size_y": 40.0, "size_z": 40.0},
                          "arrowbutton": {"diameter": 35.7, "min_x": -13.3249, "min_y": -14.2494, "min_z": -14.2494, "size_x": 26.65, "size_y": 28.5, "size_z": 28.5},
                          "redbutton": {"diameter": 35.7, "min_x": -13.3249, "min_y": -14.2494, "min_z": -14.2494, "size_x": 26.65, "size_y": 28.5, "size_z": 28.5},
                          "unknownbutton":  {"diameter": 35.7, "min_x": -13.3249, "min_y": -14.2494, "min_z": -14.2494, "size_x": 26.65, "size_y": 28.5, "size_z": 28.5}}
    
    name_to_3d_bboxes = {name: convert_bbox_3d(model_info) for name, model_info in name_to_model_info.items()}
    
    return name_to_3d_bboxes


def convert_bbox_3d(model_dict):
    """
    Converts the 3D model cuboids from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
    Args:
        model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
    Returns:
        bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.

    """
    #get infos from model dict
    min_point_x = model_dict["min_x"]
    min_point_y = model_dict["min_y"]
    min_point_z = model_dict["min_z"]
    
    size_x = model_dict["size_x"]
    size_y = model_dict["size_y"]
    size_z = model_dict["size_z"]
    
    bbox = np.zeros(shape = (8, 3))
    #lower level
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    #upper level
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
    
    return bbox


def build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights):
    """
    Builds an EfficientPose model and init it with a given weight file
    Args:
        phi: EfficientPose scaling hyperparameter
        num_classes: The number of classes
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        path_to_weights: Path to the weight file
        
    Returns:
        efficientpose_prediction: The EfficientPose model
        image_size: Integer image size used as the EfficientPose input resolution for the given phi

    """
    print("\nBuilding model...\n")
    _, efficientpose_prediction, _ = build_EfficientPose(phi,
                                                         num_classes = num_classes,
                                                         num_anchors = 9,
                                                         freeze_bn = True,
                                                         score_threshold = score_threshold,
                                                         num_rotation_parameters = 3,
                                                         print_architecture = False)
    
    print("\nDone!\n\nLoading weights...")
    efficientpose_prediction.load_weights(path_to_weights, by_name=True)
    print("Done!")
    
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    
    return efficientpose_prediction, image_size


def preprocess(image, image_size, camera_matrix, translation_scale_norm):
    """
    Preprocesses the inputs for EfficientPose
    Args:
        image: The image to predict
        image_size: Input resolution for EfficientPose
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_list: List containing the preprocessed inputs for EfficientPose
        scale: The scale factor of the resized input image and the original image

    """
    image = image[:, :, ::-1]
    image, scale = preprocess_image(image, image_size)
    camera_input = get_camera_parameter_input(camera_matrix, scale, translation_scale_norm)
    
    image_batch = np.expand_dims(image, axis=0)
    camera_batch = np.expand_dims(camera_input, axis=0)
    input_list = [image_batch, camera_batch]
    
    return input_list, scale


def get_camera_parameter_input(camera_matrix, image_scale, translation_scale_norm):
    """
    Return the input vector for the camera parameter layer
    Args:
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        image_scale: The scale factor of the resized input image and the original image
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_vector: numpy array [fx, fy, px, py, translation_scale_norm, image_scale]

    """
    #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
    input_vector = np.zeros((6,), dtype = np.float32)
    
    input_vector[0] = camera_matrix[0, 0]
    input_vector[1] = camera_matrix[1, 1]
    input_vector[2] = camera_matrix[0, 2]
    input_vector[3] = camera_matrix[1, 2]
    input_vector[4] = translation_scale_norm
    input_vector[5] = image_scale
    
    return input_vector


def postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold):
    """
    Filter out detections with low confidence scores and rescale the outputs of EfficientPose
    Args:
        boxes: numpy array [batch_size = 1, max_detections, 4] containing the 2D bounding boxes
        scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
        labels: numpy array [batch_size = 1, max_detections] containing class label
        rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
        scale: The scale factor of the resized input image and the original image
        score_threshold: Minimum score threshold at which a prediction is not filtered out
    Returns:
        boxes: numpy array [num_valid_detections, 4] containing the 2D bounding boxes
        scores: numpy array [num_valid_detections] containing the confidence scores
        labels: numpy array [num_valid_detections] containing class label
        rotations: numpy array [num_valid_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [num_valid_detections, 3] containing the translation vectors

    """
    boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
    # correct boxes for image scale
    boxes /= scale
    #rescale rotations
    rotations *= math.pi
    #filter out detections with low scores
    indices = np.where(scores[:] > score_threshold)
    # select detections
    scores = scores[indices]
    boxes = boxes[indices]
    rotations = rotations[indices]
    translations = translations[indices]
    labels = labels[indices]
    
    return boxes, scores, labels, rotations, translations

def parse_data(labels, rotations, translations, class_to_name):

    capture_translations = {"2-slot": [0, 0, 0],
                       "3-slot": [0, 0, 0],
                       "mushroombutton": [0, 0, 0],
                       "arrowbutton": [0, 0, 0],
                       "redbutton": [0, 0, 0],
                       "unknownbutton": [0, 0, 0]}

    capture_rotations = {"2-slot": [0, 0, 0],
                       "3-slot": [0, 0, 0],
                       "mushroombutton": [0, 0, 0],
                       "arrowbutton": [0, 0, 0],
                       "redbutton": [0, 0, 0],
                       "unknownbutton": [0, 0, 0]}

    i = 0
    for label in labels:
        capture_translations[class_to_name[label]] = translations[i].tolist()
        capture_rotations[class_to_name[label]] = rotations[i].tolist()
        i += 1

    return {"translations": capture_translations, "rotations": capture_rotations}


if __name__ == '__main__':
    main()
