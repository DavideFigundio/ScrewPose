'''
Script for running live inference on an Azure Kinect camera.
'''

import cv2
import numpy as np
import os
import math

import tensorflow as tf
import pykinect_azure as pykinect

from model import build_EfficientPose
from utils import preprocess_image
from utils.visualization import draw_detections


def main():
    """
    Run EfficientPose in inference mode live on azure kinect.
    
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()

    #input parameter
    phi = 0
    path_to_weights = "./weights/assembly_150_epochs.h5"
    # save_path = "./predictions/occlusion/" #where to save the images or None if the images should be displayed and not saved
    save_path = None
    image_extension = ".jpg"
    class_to_name = {0: "screw", 1: "workpiece"} 
    #class_to_name = {0: "driller"} #Linemod use a single class with a name of the Linemod objects
    score_threshold = 0.1
    translation_scale_norm = 1000.0
    draw_bbox_2d = False
    draw_name = False
    #you probably need to replace the linemod camera matrix with the one of your webcam
    camera_matrix = get_camera_matrix()
    name_to_3d_bboxes = get_linemod_3d_bboxes()
    class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
    
    num_classes = len(class_to_name)
    
    #build model and load weights
    model, image_size = build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights)
    
    print("\nSetting up Azure Kinect...\n")

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

	# Modify camera configuration
    device_config = pykinect.default_configuration
	# print(device_config)

	# Start device
    device = pykinect.start_device(config=device_config)
    
    #inferencing
    print("\nStarting inference...\n")
    cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)

    i = 0
    while True:

		# Get capture
        capture = device.update()

		# Get the color image from the capture
        ret, image = capture.get_color_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        if not ret:
            continue

        original_image = image.copy()
        
        #preprocessing
        input_list, scale = preprocess(image, image_size, camera_matrix, translation_scale_norm)
        
        #predict
        boxes, scores, labels, rotations, translations = model.predict_on_batch(input_list)
        
        #postprocessing
        boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold)
        
        # print('Number of boxes detected:' + str(len(boxes)))
        print(translations)

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
		
		# Plot the image
        cv2.imshow("Color Image",original_image)
		
		# Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break
    #release webcam and close windows
    cv2.destroyAllWindows()
    

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)



def get_camera_matrix():
    """
    Returns:
        The Azure Kinect 3x3 camera matrix

    """
    return np.array([[612.64, 0., 638.02], [0., 612.36, 367.65], [0., 0., 1.]], dtype = np.float32)


def get_linemod_3d_bboxes():
    """
    Returns:
        name_to_3d_bboxes: Dictionary with the Linemod and Occlusion 3D model names as keys and the cuboids as values

    """
    name_to_model_info = {"screw":          {"diameter": 37.2738, "min_x": -5.7735, "min_y": -5.0, "min_z": -17, "size_x": 11.547, "size_y": 10.0, "size_z": 34.0},
                            "workpiece":    {"diameter": 51.9615, "min_x": -15.0, "min_y": -15.0, "min_z": -15.0, "size_x": 30.0, "size_y": 30.0, "size_z": 30.0}}
        
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


if __name__ == '__main__':
    main()
