from cv2 import initCameraMatrix2D
import numpy as np
import cv2

def create_ICP(iterations):
    return cv2.ppf_match_3d.ICP(iterations)

def pose_refinement(ICP, capture, labels, rotation_estimates, translation_estimates, bboxes, class_to_name_dict, name_to_model_dict, camera_matrix, visualizer):
    """
    Uses ICP to refine the given poses.
    Args:
        pointcloud - numpy array of (x, y, z) tuples of 16-bit signed integers representing point coordinates in mm from the camera reference
        translation_estimates - initial estimates of translations directly from EfficientPose, in mm
        rotation_estimates - initial estimates of rotations directly from EfficientPose, represented as a Rodrigues vector
    """

    rotations = np.empty((0, 3), dtype=np.float32)
    translations = np.empty((0, 3), dtype=np.float32)
    #ret, pointcloud = cv2.ppf_match_3d.computeNormalsPC3d(pointcloud, 15, True, np.array([0., 0., 0.]))
    #pointcloud = np.append(pointcloud.astype("float32"), np.zeros((len(pointcloud), 3), dtype=np.float32), axis=1)
    depth_image = rescale_depth_image(capture)

    i = 0  
    for label in labels:
        print(label)
        pose = cv2.ppf_match_3d.Pose3D()
        pose.updatePose(make4x4matrix(rotation_estimates[i], translation_estimates[i]))
        
        if class_to_name_dict[label] == "M8x50":
            print("Applying ICP...")
            pointcloud = create_pointcloud(depth_image, bboxes[i], camera_matrix)
            model = name_to_model_dict[class_to_name_dict[label]]
            posediff = ICP.registerModelToScene(model, pointcloud, [pose])
            pose = posediff[1][0]
            savePointcloud(pointcloud, "test.ply")
            visualizer(pointcloud)
        
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

def rescale_depth_image(capture):
    depth = capture.get_depth_image_object() 

    depth = capture.camera_transform.depth_image_to_color_camera(depth)
    _, depth_image = depth.to_numpy()

    return depth_image

def create_pointcloud(depth_image, bbox, camera_matrix):
    """
    Creates a pointcloud from a depth image.
    Args:
        depth_image - numpy matrix containing for each pixel the depth in mm
        bbox - the area inside the depth image to be considered. 1x4 numpy array
                specifying the pixel coordinates of a rectangular bounding box
                [x1, y1, x2, y2]
        camera_matrix - 3x3 numpy array containing intrinsic parameters for the depth image.
    Returns:
        pointcloud - Nx3 numpy array that contains x, y, z coordinates in mm for each of N points.
    """
    pointcloud = np.empty((0, 3), dtype=np.float32)
    inverse = np.linalg.inv(camera_matrix)
    bbox = bbox.astype(np.intc)

    for x in range(bbox[0], bbox[2] + 1):
        for y in range(bbox[1], bbox[3] + 1):
            z = depth_image[y][x]
            if z != 0:
                pixel_coords = np.array([x, y, 1], dtype=np.float32)
                coords = np.matmul(inverse, pixel_coords)*z

                pointcloud = np.append(pointcloud, np.expand_dims(coords, axis=0), axis=0)

    return pointcloud

def savePointcloud(xyz_points, filename):

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    
    fid = open(filename,'w')
    fid.write("ply\n")
    fid.write("format ascii 1.0\n")
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
