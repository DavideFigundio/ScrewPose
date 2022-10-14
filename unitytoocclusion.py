'''
Script for converting the output of a Unity Perception synthetic dataset to a format that is useable
with EfficentPose for training an occlusion linemod model.
'''

from math import ceil
import yaml
import json
import os
from os.path import join

def main():
    datalocationpath = './datasets/ScrewPose/data/'
    numberOfSamples = 10000
    trainamount = 9000
    samplesPerCaptureFile = 150
    nameToIdDict = {"M8x50": 1, "M8x25": 2, "M8x16": 3, "M6x30": 4}

    Cam_intrinsic = [640.0, 0., 640.0, 0., 640.0, 360.0, 0., 0., 1.]

    deletePreviousGT(datalocationpath)
    
    createLabels(datalocationpath, samplesPerCaptureFile, numberOfSamples, Cam_intrinsic, nameToIdDict)

    createSplits(datalocationpath, trainamount, numberOfSamples)

    #renameImages(datalocationpath, numberOfSamples)

    print("Done!")

def createLabels(datalocationpath, samplesPerCaptureFile, numberOfSamples, Cam_intrinsic, nameToIdDict):
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
            data = {i:{'cam_K':Cam_intrinsic, 'depth_scale':1.0}}
            yaml.dump(data, ymlfile)

def convertQuaternionToMatrix(qx, qy, qz, qw):
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
    for i in range(0, numberOfSamples):
        if i<10:
            os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/000' + str(i) + '.png')
            os.rename(datalocationpath + "merged_masks/segmentation_" + str(i + 2) + ".png", datalocationpath + 'merged_masks/000' + str(i) + '.png')
        elif i<100:
            os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/00' + str(i) + '.png')
            os.rename(datalocationpath + "merged_masks/segmentation_" + str(i + 2) + ".png", datalocationpath + 'merged_masks/00' + str(i) + '.png')
        elif i<1000:
            os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/0' + str(i) + '.png')
            os.rename(datalocationpath + "merged_masks/segmentation_" + str(i + 2) + ".png", datalocationpath + 'merged_masks/0' + str(i) + '.png')     
        else:
            os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/' + str(i) + '.png')
            os.rename(datalocationpath + "merged_masks/segmentation_" + str(i + 2) + ".png", datalocationpath + 'merged_masks/' + str(i) + '.png')

def deletePreviousGT(dirpath):
    os.remove(join(dirpath, "gt.yml"))
    os.remove(join(dirpath, "info.yml"))
    os.remove(join(dirpath, "test.txt"))
    os.remove(join(dirpath, "train.txt"))
    os.remove(join(dirpath, "valid_poses/1.txt"))
    os.remove(join(dirpath, "valid_poses/2.txt"))

class ObjectData:

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