import cv2      # Requires opencv-contrib-python
from os import listdir
from os.path import join
import numpy as np
import json
from pyquaternion import Quaternion

# Takes a directory of images and for each image saves all poses of a chosed ArUco marker in a json file for use in Unity.
def main():
    dirpath = "../background/"  # Path to image directory
    verpath = "../verification/"
    jsonData = "poses.json"     # Name of json file
    markerLength = 0.041        # Marker size [m]
    markerID = 23               # Reference marker ID

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arcucoParams = cv2.aruco.DetectorParameters_create()

    # Camera parameters are for Azure Kinect @720p resolution
    cameraMatrix = np.array([[612.6460571289062, 0., 638.0296020507812], [0., 612.36376953125, 367.6560363769531], [0., 0., 1.]], dtype = np.float32)
    distortionMatrix = np.array([0.5059323906898499, -2.6153206825256348, 0.000860791013110429, -0.0003529376117512584, 1.4836950302124023], dtype = np.float32)

    posedict = dict()
    
    images = [f for f in listdir(dirpath) if f.endswith(".png")]
    for imgname in images:
        print("Currently reading: " + imgname, end='\r') #
        image = cv2.imread(join(dirpath, imgname))
        (corners, ids, _) = cv2.aruco.detectMarkers(image, dictionary, parameters=arcucoParams)

        if ids.any():
            rvects, tvects, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distortionMatrix)

            for i in range(0, len(ids)):
                if ids[i] == markerID:
                    #cv2.drawFrameAxes(image, cameraMatrix, distortionMatrix, rvects[i], tvects[i], 0.1)
                    posedict[imgname] = {"rotation": RodriguezToUnityQuaternion(rvects[i][0]), "translation": {'x': tvects[i][0][0], 'y': -tvects[i][0][1], 'z': tvects[i][0][2]}}
        
        #cv2.imwrite(join(verpath, imgname), image)
    
    print("Saving to json...             ")
    with open(join(dirpath, jsonData), 'w') as file:
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