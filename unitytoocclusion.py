'''
Script for converting the output of a Unity Perception synthetic dataset to a format that is useable
with EfficentPose for training an occlusion linemod model.
'''

from math import ceil
import yaml
import json
import os
import cv2
import numpy as np
from os.path import join

def main():
    datalocationpath = './datasets/ButtonPose/data/'
    numberOfSamples = 20000
    trainamount = 18000
    samplesPerCaptureFile = 150

    #nameToIdDict = {"M8x50": 1, "M8x25": 2, "M8x16": 3, "M6x30": 4} # ScrewPose
    nameToIdDict = {"2-slot": 1, "3-slot": 2, "mushroombutton": 3, "arrowbutton": 4, "redbutton": 5, "unknownbutton": 6} # ButtonPose
    nameToMaskDict = {"2-slot": 42, "3-slot": 84, "mushroombutton": 126, "arrowbutton": 168, "redbutton": 210, "unknownbutton": 255}

    #Cam_intrinsic = [905., 0., 640., 0., 905., 360., 0.0, 0., 1.] # Realsense parameters
    Cam_intrinsic = [640.0, 0., 640.0, 0., 640.0, 360.0, 0., 0., 1.] # Azure paremeters

    deletePreviousGT(datalocationpath, len(nameToIdDict))

    occlusionDict = loadOcclusionData(join(datalocationpath, "occlusions.txt"))
    
    createLabels(datalocationpath, samplesPerCaptureFile, numberOfSamples, Cam_intrinsic, nameToIdDict, occlusionDict)

    createSplits(datalocationpath, trainamount, numberOfSamples)

    renameImages(datalocationpath, numberOfSamples)

    if occlusionDict:
        correctMasks(datalocationpath, occlusionDict, nameToMaskDict)

    print("All tasks finished.                              ")

def createLabels(datalocationpath, samplesPerCaptureFile, numberOfSamples, Cam_intrinsic, nameToIdDict, occlusionDict):
    print("Creating ground truth labels...")

    totalCaptures = ceil(numberOfSamples/samplesPerCaptureFile)
    imageData = {name : ObjectData(nameToIdDict[name]) for name in nameToIdDict.keys()}

    j=0
    with open(join(datalocationpath, 'gt.yml'), 'w') as ymlfile:
        for i in range(0, totalCaptures):
            captureFileNumber = ''
            if i<10:
                captureFileNumber = '00' + str(i)
            elif i<100:
                captureFileNumber = '0' + str(i)
            else:
                captureFileNumber = str(i)

            with open(join(datalocationpath, "unitydata/captures_" + captureFileNumber + ".json"), 'r') as file:
                jsonData = json.load(file)    

                for i in range(0, len(jsonData['captures'])):
                    print("Progress: " + str(j) + "/" + str(numberOfSamples), end='\r')

                    for annotation in jsonData['captures'][i]['annotations']:

                        if annotation['id'] == 'bounding box':

                            for bboxdata in annotation['values']:
                                name = bboxdata['label_name']
                                imageData[name].bbox = [bboxdata['x'], bboxdata['y'], bboxdata['width'], bboxdata['height']]
                                imageData[name].present = True

                        elif annotation['id'] == 'bounding box 3D':

                            for bboxdata in annotation['values']:
                                name = bboxdata['label_name']

                                translationdata = bboxdata['translation']
                                rotationdata = bboxdata['rotation']

                                imageData[name].translation = [1000*translationdata['x'], -1000*translationdata['y'], 1000*translationdata['z']]
                                imageData[name].rotation = convertQuaternionToMatrix(rotationdata['x'], rotationdata['y'], rotationdata['z'], rotationdata['w'])

                    if j in occlusionDict.keys():
                        if imageData[occlusionDict[j]].present:
                            imageData["unknownbutton"].present = True
                            imageData["unknownbutton"].rotation = imageData[occlusionDict[j]].rotation
                            imageData["unknownbutton"].translation = imageData[occlusionDict[j]].translation
                            imageData["unknownbutton"].bbox = imageData[occlusionDict[j]].bbox

                            imageData[occlusionDict[j]].clear()


                    datalist = []
                    
                    for key in imageData.keys():
                        if imageData[key].present:
                            datalist.append({'cam_R_m2c':imageData[key].rotation, 'cam_t_m2c':imageData[key].translation, 'obj_bb':imageData[key].bbox, 'obj_id':imageData[key].ID})
                            with open(join(datalocationpath, imageData[key].valid_poses_file), "a") as file:
                                file.write(str(j) + '\n')
                            imageData[key].clear()
        
                    yaml.dump({j:datalist}, ymlfile)
                    j += 1

    print("Creating info labels...")
    with open(datalocationpath + 'info.yml', 'w') as ymlfile:
        for i in range(0, numberOfSamples):
            print("Progress: " + str(i) + "/" + str(numberOfSamples), end='\r')
            data = {i:{'cam_K':Cam_intrinsic, 'depth_scale':1.0}}
            yaml.dump(data, ymlfile)

def convertQuaternionToMatrix(qx, qy, qz, qw):
    # Transforms a quaternion rotation to its equivalent matrix form.

    qw = -qw
    qy = -qy
    R11 = 1 - 2 * (qy * qy + qz * qz) 
    R12 = 2 * (qx * qy - qz * qw)
    R13 = 2 * (qx * qz + qy * qw)
    R21 = 2 * (qx * qy + qz * qw)
    R22 = 1 - 2 * (qx * qx + qz * qz)
    R23 = 2 * (qy * qz - qx * qw)
    R31 = 2 * (qx * qz - qy * qw)
    R32 = 2 * (qy * qz + qx * qw)
    R33 = 1 - 2 * (qx * qx + qy * qy)

    R = [R11, R12, R13, R21, R22, R23, R31, R32, R33]

    return R

def createSplits(datalocationpath, trainamount, numberOfSamples):
    # Creates the files specifying what files to use for training and what
    # files to use for testing the model.

    print('Creating test/train splits...')
    with open(datalocationpath + 'train.txt', 'w') as file:
        for i in range(0, trainamount):
            if i<10:
                file.write('000' + str(i) + '\n')
            elif i<100:
                file.write('00' + str(i) + '\n')
            elif i<1000:
                file.write('0' + str(i) + '\n')
            else:
                file.write(str(i)+'\n')

    with open(datalocationpath + 'test.txt', 'w') as file:
        for i in range(trainamount, numberOfSamples):
            if i<10:
                file.write('000' + str(i) + '\n')
            elif i<100:
                file.write('00' + str(i) + '\n')
            elif i<1000:
                file.write('0' + str(i) + '\n')
            else:
                file.write(str(i)+'\n')

def renameImages(datalocationpath, numberOfSamples):
    print('Renaming training images...')
    if(os.path.exists(join(datalocationpath, "rgb/rgb_2.png"))):
        for i in range(0, numberOfSamples):
            print("Progress: " + str(i) + "/" + str(numberOfSamples), end='\r')
            os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/' + str(i) + '.png')
            os.rename(datalocationpath + "merged_masks/segmentation_" + str(i + 2) + ".png", datalocationpath + 'merged_masks/' + str(i) + '.png')
    else:
        print("Found already renamed training images.")

def deletePreviousGT(dirpath, objectnumber):
    # Eliminates previous versions of ground truths in the given path.

    if(os.path.exists(join(join(dirpath, "gt.yml")))):
        os.remove(join(dirpath, "gt.yml"))

    if(os.path.exists(join(join(dirpath, "info.yml")))):  
        os.remove(join(dirpath, "info.yml"))

    if(os.path.exists(join(join(dirpath, "test.txt")))):
        os.remove(join(dirpath, "test.txt"))

    if(os.path.exists(join(join(dirpath, "train.txt")))):
        os.remove(join(dirpath, "train.txt"))

    if(os.path.exists(join(join(dirpath, "valid_poses")))):
        valid_pose_dir = join(dirpath, "valid_poses")
        valid_pose_files = os.listdir(valid_pose_dir)
        filtered_files  = [file for file in valid_pose_files if file.endswith(".txt")]
        if len(filtered_files) > 0:
            for file in filtered_files:
                os.remove(join(valid_pose_dir, file))
    else:
        os.mkdir("valid_poses")

    for i in range(1, objectnumber + 1):
        with open(join(dirpath, "valid_poses/" + str(i) + ".txt"), 'w') as f:
            pass

def loadOcclusionData(filepath):
    # Loads information on occlusions from the given .txt file, if present.

    print("Loading occlusion data..")
    occlusionDict = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            for line in file:
                (key, val) = line.split()
                occlusionDict[int(key)] = val
    else:
        print("No occlusion data found.")
    
    return occlusionDict

def correctMasks(basepath, occlusionData, nameToMaskDict):
    # Corrects masks using occlusion data.

    print("Correcting masks with occlusion data...")

    for key in occlusionData.keys():
        imgname = str(key) + ".png"
        print("Currently correcting: " + imgname, end='\r')

        imagepath = join(basepath, "merged_masks/" + imgname)
        maskValue = nameToMaskDict[occlusionData[key]]

        image = cv2.imread(imagepath)
        low = np.array([maskValue - 5, maskValue - 5, maskValue - 5])
        high = np.array([maskValue + 5, maskValue + 5, maskValue + 5])

        mask = cv2.inRange(image, low, high)
        image[mask > 0] = (255, 255, 255)
        cv2.imwrite(imagepath, image)



class ObjectData:
    # Class used to simplify storing object data and creating dictionaries.

    def __init__(self, ID):
        self.ID = ID
        self.present = False
        self.bbox = None
        self.translation = None
        self.rotation = None
        self.valid_poses_file = "valid_poses/" + str(ID) + ".txt"

    def clear(self):
        self.present = False
        self.bbox = None
        self.translation = None
        self.rotation = None

if __name__ == "__main__":
    main()