'''
Script for running live inference on an Azure Kinect camera.
'''

import cv2
import numpy as np
import os
import math
import struct
import tensorflow as tf
import pykinect_azure as pykinect
from plyfile import PlyData, PlyElement
from pykinect_azure.utils import Open3dVisualizer

import icp_refinement as icp
from model import build_EfficientPose
from utils import preprocess_image
from utils.visualization import draw_detections

DO_ICP = False
VISUALIZE_POINTCLOUD = False
CLASS_TO_NAME = {0 : "M8x50", 1 : "M8x25", 2 : "M8x16", 3 : "M6x30"}

def main():
    """
    Run EfficientPose in inference mode live on Azure Kinect.
    
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()

    #input parameters
    phi = 0
    path_to_weights = "./weights/screwpose_100_epochs.h5"
    path_to_models = "./models/"
    class_to_name = CLASS_TO_NAME
    score_threshold = 0.5
    translation_scale_norm = 1000.0     # conversion factor from m to mm
    draw_bbox_2d = False
    draw_name = False
    
    name_to_3d_bboxes = get_3d_bboxes()
    class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
    num_classes = len(class_to_name)
    
    name_to_models = load_3D_models(class_to_name, path_to_models)

    ICP = icp.create_ICP(100)

    #build model and load weights
    model, image_size = build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights)
    print("\nSetting up Azure Kinect...\n")

    visualizer = Open3dVisualizer()

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

	# Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P

    if DO_ICP:
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
        device_config.synchronized_images_only = True
        
    

	# Start device
    device = pykinect.start_device(config=device_config)

    # Getting intrinsic parameters
    print_params = True
    camera_matrix, dist, depthmat = get_camera_params(device.calibration, print_params)
    
    cv2.namedWindow('ScrewPose', cv2.WINDOW_NORMAL)

    #inferencing
    print("\nStarting inference...\n")

    #visualizer = Open3dVisualizer()

    while True:

		# Get capture
        capture = device.update()

		# Get the color image from the capture
        ret, image = capture.get_color_image()
        
        if not ret:
            continue

        
        # Removing alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Undistorting the imgae
        image = cv2.undistort(image, camera_matrix, dist)
        
        # Preprocessing
        original_image = image.copy()
        input_list, scale = preprocess(image, image_size, camera_matrix, translation_scale_norm)
        
        # Pose inference with EfficientPose
        boxes, scores, labels, rotations, translations = model.predict_on_batch(input_list)
        
        # Postprocessing
        boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold)

        # Pose refinement with ICP
        if DO_ICP and translations.any():
            _, pointcloud = capture.get_pointcloud()
            
            # print(type(pointcloud))
            # print(pointcloud)
            # savePointcloud(pointcloud, "pointcloud.ply")
            # break

            pointcloud = icp.crop_pointcloud(pointcloud, translations[0], 40)
            visualizer(pointcloud)
            #visualizer(cropped_pc)
            rotations, translations = icp.pose_refinement(ICP, pointcloud, labels, rotations, translations, class_to_name, name_to_models)

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
        
    #release webcam and close windows
    cv2.destroyAllWindows()
    

def allow_gpu_growth_memory():
    """image_size
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)

def get_camera_params(calibration, print_params=False):
    """
    Gets camera parameters from current Azure calibration settings.
    Args:
        calibration: A calibration object from an Azure Kinect camera.
        print: bool, if true prints the calibration parameters when executed.
    Returns:
        mat: 3x3 camera intrinsic matrix of the type:
            |fx  0   cx|
            |0   fy  cy|
            |0   0   1 |
        dist: camera distortion parameters in the form:
            [k1, k2, p1, p2, k3, k4, k5, k6]
    """
    if print_params:
        print(calibration)

    params = calibration.color_params
    depth_params = calibration.depth_params

    mat = np.array(calibration.get_matrix("color"), dtype=np.float32)

    dist = np.array([params.k1, params.k2, params.p1, params.p2, params.k3, params.k4, params.k5, params.k6], dtype = np.float32)

    depthmat = np.array(calibration.get_matrix("depth"), dtype= np.float32)

    return mat, dist, depthmat

def get_3d_bboxes():
    """
    Returns:
        name_to_3d_bboxes: Dictionary with the Linemod and Occlusion 3D model names as keys and the cuboids as values

    """
    name_to_model_info = {"M8x50":  {"diameter": 58.9428, "min_x": -6.5, "min_y": -6.4723, "min_z": -29.0, "size_x": 13.0, "size_y": 12.9445, "size_z": 58.0},
                          "M8x25":  {"diameter": 34.6302, "min_x": -6.5, "min_y": -6.5, "min_z": -15.5, "size_x": 13.0, "size_y": 13.0, "size_z": 33.0},
                          "M8x16":  {"diameter": 21.4, "min_x": -7.0, "min_y": -6.9837, "min_z": -10.2, "size_x": 14.0, "size_y": 13.9674, "size_z": 20.4},
                          "M6x30":  {"diameter": 37.2738, "min_x": -5.7735, "min_y": -5.0, "min_z": -17.0, "size_x": 11.547, "size_y": 10.0, "size_z": 34.0}}
    
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

def savePCtoPLY(points, filepath):
    if len(points)>0:
        with open("pointcloud.txt", "x") as file:
            for point in points:
                file.write(str(point))
        ply_vertices = np.empty(len(points), dtype=[("vertices", np.float32, (3, ))])
        ply_vertices["vertices"] = points
        pc = PlyElement.describe(ply_vertices, "vertex")
        pc = PlyData([pc])
        with open(filepath, 'wb') as f:
            pc.write(f)

def load_3D_models(class_to_name_dict, path_to_models):
    """
    Function that loads the 3d models from a directory. Models must be in the .ply format and each model must be named in the format "<name>.ply",
    where <name> is the name it has inside the class_to_name dictionary.
    Args:
        class_to_name_dict - dictionary that associates each class to the corresponding model name
        path_to_models - location where the .ply files are stored
    Returns:
        name_to_model_dict - dictionary that associates each name to a pointcloud that represents its 3D model
    """

    name_to_model_dict = {}

    for key in class_to_name_dict.keys():
        model_data = PlyData.read(os.path.join(path_to_models, class_to_name_dict[key] + ".ply"))
                                  
        vertex = model_data['vertex']
        points = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis = -1).astype(np.float32)
        
        #points = np.append(points.astype("float32"), np.zeros((len(points), 3), dtype=np.float32), axis=1)
        points = points.astype("float32")
        name_to_model_dict[class_to_name_dict[key]] = points

    return name_to_model_dict

def savePointcloud(xyz_points, filename):

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    
    fid = open(filename,'w')
    fid.write("ply\n")
    fid.write("format binary_little_endian 1.0\n")
    #fid.write("element vertex %d\n"%xyz_points.shape[0]")
    fid.write("element vertex "+ str(xyz_points.shape[0]) + "\n")
    fid.write("property float x\n")
    fid.write("property float y\n")
    fid.write("property float z\n")
    fid.write("end_header\n")    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        if xyz_points[i, 0] == 0 and xyz_points[i, 1] == 0 and xyz_points[i, 0] == 0:
            continue
        fid.write(str(xyz_points[i,0]) + " " + 
                  str(xyz_points[i,1]) + " " +
                  str(xyz_points[i,2]) + "\n")
    fid.close()


if __name__ == '__main__':
    main()
