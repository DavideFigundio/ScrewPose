import cv2      # Requires opencv-contrib-python
from os.path import join
import numpy as np
import json
from pyquaternion import Quaternion

UNDISTORT = False
SAVE_VERIFICATION = True

# Takes a directory of images and for each image saves all poses of a chosed ArUco marker in a json file for use in Unity.
def main():
    dirpath = "../background/realsense"  # Path to image directory
    # dirpath = "../background/azure"

    distdirpath = "../background_undistorted/"
    verpath = "../verification/"
    jsonFile = "poses_rs.json"     # Name of json file
    # jsonFile = "poses_azure.json"
    markerLength = 0.04         # Marker size [m]
    markerID = 23               # Reference marker ID

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arcucoParams = cv2.aruco.DetectorParameters_create()

    # Camera parameters for Azure Kinect @720p
    # cameraMatrix = np.array([[612.6460571289062, 0., 638.0296020507812], [0., 612.36376953125, 367.6560363769531], [0., 0., 1.]], dtype = np.float32)
    # distortionMatrix = np.array([0.5059323906898499, -2.6153206825256348, 0.000860791013110429, -0.0003529376117512584, 1.4836950302124023, 0.3840336799621582, -2.438732385635376, 1.4119256734848022], dtype = np.float32)

    # Camera parameters for Intel Realsense @ 720p
    cameraMatrix = np.array([[904.9464, 0., 638.6059], [0.,  905.1834, 363.14758], [0., 0., 1.]], dtype = np.float32)
    distortionMatrix = np.array([0., 0., 0., 0., 0.], dtype = np.float32)

    posedict = dict()
    total = 950
    for i in range(total):
        imgname = str(i) + ".png"
        print("Currently reading: " + imgname, end='\r') #
        image = cv2.imread(join(dirpath, imgname))
        (corners, ids, _) = cv2.aruco.detectMarkers(image, dictionary, parameters=arcucoParams)

        original_image = image.copy()

        if ids.any():
            rvects, tvects, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distortionMatrix)

            for j in range(0, len(ids)):
                if ids[j] == markerID:
                    if SAVE_VERIFICATION:
                        cv2.drawFrameAxes(image, cameraMatrix, distortionMatrix, rvects[j], tvects[j], 0.1)
                    posedict[i] = {"rotation": RodriguezToUnityQuaternion(rvects[j][0]), "translation": {'x': tvects[j][0][0], 'y': -tvects[j][0][1], 'z': tvects[j][0][2]}}
        
        if UNDISTORT:
            dst = cv2.undistort(original_image, cameraMatrix, distortionMatrix, None)
            cv2.imwrite(join(distdirpath, imgname), dst)

        if SAVE_VERIFICATION:
            cv2.imwrite(join(verpath, imgname), image)
    
    print("Saving to json...             ")
    with open(join(dirpath, jsonFile), 'w') as file:
        json.dump(posedict, file)

    print("Done!")

def RodriguezToUnityQuaternion(rotation):
    #Function that transforms a rotation from Rodriguez to Unity's LHS quaternion
    rotMat, _ = cv2.Rodrigues(rotation)
    rotQuat = Quaternion(matrix=rotMat)
    elements = rotQuat.elements
    return {'x':elements[1], 'y':-elements[2], 'z': elements[3], 'w': -elements[0]}

if __name__ == "__main__":
    main()