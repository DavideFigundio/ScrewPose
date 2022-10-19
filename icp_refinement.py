from cv2 import initCameraMatrix2D
import numpy as np
import cv2

def create_ICP(iterations):
    return cv2.ppf_match_3d.ICP(iterations)

def pose_refinement(ICP, pointcloud, labels, rotation_estimates, translation_estimates, class_to_name_dict, name_to_model_dict):
    """
    Uses ICP to refine the given poses.
    Args:
        pointcloud - numpy array of (x, y, z) tuples of 16-bit signed integers representing point coordinates in mm from the camera reference
        translation_estimates - initial estimates of translations directly from EfficientPose, in mm
        rotation_estimates - initial estimates of rotations directly from EfficientPose, represented as a Rodrigues vector
    """

    rotations = np.empty((0, 3), dtype=np.float32)
    translations = np.empty((0, 3), dtype=np.float32)
    pointcloud = pointcloud.astype("float32")
    #ret, pointcloud = cv2.ppf_match_3d.computeNormalsPC3d(pointcloud, 15, True, np.array([0., 0., 0.]))
    #pointcloud = np.append(pointcloud.astype("float32"), np.zeros((len(pointcloud), 3), dtype=np.float32), axis=1)

    i = 0  
    for label in labels:
        model = name_to_model_dict[class_to_name_dict[label]]
        pose = cv2.ppf_match_3d.Pose3D()
        pose.updatePose(make4x4matrix(rotation_estimates[i], translation_estimates[i]))
        
        if class_to_name_dict[label] == "M8x50":
            print("Applying ICP...")
            posediff = ICP.registerModelToScene(model, pointcloud, [pose])
            pose = posediff[1][0]
        
        rot, trans = unmake4x4matrix(pose.pose)
        rotations = np.append(rotations, rot, axis=0)
        translations = np.append(translations, trans, axis=0)
        
        i += 1

    return rotations, translations

def crop_pointcloud(pointcloud, center, size):
    cropped_pc = np.empty((0, 3), dtype=np.float32)

    for point in pointcloud:
        if abs(point[0] - center[0]) >= size:
            continue
        if abs(point[1] - center[1]) >= size:
            continue
        if abs(point[2] - center[2]) >= size:
            continue

        cropped_pc = np.append(cropped_pc, np.expand_dims(point, axis=0), axis=0)
    
    return cropped_pc

def make4x4matrix(rotation, translation):
    """
    Function that turns a rotation (Rodrigues angles) and translation into a
    homogeneous transform.
    Args:
        rotation - 1x3 numpy array containing the components of the rotation vector
        translation - 1x3 numpy array containing the components of the translation vector
    Returns
        mat - 4x4 numpy array containing the homogeneous transform matrix
    """
    rotMat, _ = cv2.Rodrigues(rotation)

    mat = np.append(rotMat, np.transpose(np.expand_dims(translation, axis=0)), axis=1)
    mat = np.append(mat, np.array([[0., 0., 0., 1.]], dtype=np.float32), axis=0)

    #return np.transpose(mat)
    return mat

def unmake4x4matrix(matrix):
    """
    Function that calculates rotation and translation vectors from a homogeneous transform matrix.
    Args:
        matrix - 4x4 numpy array containing the homogeneous transform
    Returns:
        rotation - 1x3 numpy array containing the components of the rotation vector (Rodrigues angles)
        translation - 1x3 numpy array containing the components of the translation vector
    """
    #matrix = np.transpose(matrix)
    rotmat = np.transpose(np.transpose(matrix[:3])[:3])
    translation = np.expand_dims(np.transpose(matrix[:3])[3], axis=0)
    rotation, _ = cv2.Rodrigues(rotmat)

    return np.transpose(rotation), translation
