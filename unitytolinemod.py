'''
Script for converting the output of a Unity Perception synthetic dataset to a format that is useable
with EfficentPose for training.
'''

from ctypes import sizeof
from math import ceil
import yaml
import json
import numpy as np
import math
import os

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

datalocationpath = 'C:/Users/dfigu/Documents/Projects/Tesi/Network/datasets/ScrewDataset/data/01/'
numberOfSamples = 10000
trainamount = 9500
samplesPerCaptureFile = 150
totalCaptures = ceil(numberOfSamples/samplesPerCaptureFile)

j=0
Cam_intrinsic = [633.96, 0.0, 320.0, 0.0, 633.96, 240.0, 0.0, 0.0, 1.0]
print("Creating ground truth labels...")
with open(datalocationpath + 'gt.yml', 'w') as ymlfile:
    for i in range(0, totalCaptures):
        captureFileNumber = ''
        if i<10:
            captureFileNumber = '00' + str(i)
        elif i<100:
            captureFileNumber = '0' + str(i)
        else:
            captureFileNumber = str(i)

        with open(datalocationpath + "unitydata/captures_" + captureFileNumber + ".json", 'r') as file:
            jsonData = json.load(file)    

            for i in range(0, len(jsonData['captures'])):
                for annotation in jsonData['captures'][i]['annotations']:
                    if annotation['id'] == 'bounding box':
                        bboxdata = annotation['values'][0]
                        bbox = [bboxdata['x'], bboxdata['y'], bboxdata['width'], bboxdata['height']]
                    
                    elif annotation['id'] == 'bounding box 3D':
                        translationdata = annotation['values'][0]['translation']
                        translation = [1000*translationdata['x'], -1000*translationdata['y'], 1000*translationdata['z']]

                        rotationdata = annotation['values'][0]['rotation']
                        rotation = convertQuaternionToMatrix(rotationdata['x'], rotationdata['y'], rotationdata['z'], rotationdata['w'])

                captureData = {j:[{'cam_R_m2c':rotation, 'cam_t_m2c':translation, 'obj_bb':bbox, 'obj_id':1}]}
                yaml.dump(captureData, ymlfile)
                j += 1

print("Creating info labels...")
with open(datalocationpath + 'info.yml', 'w') as ymlfile:
    for i in range(0, numberOfSamples):
        data = {i:{'cam_K':Cam_intrinsic, 'depth_scale':1.0}}
        yaml.dump(data, ymlfile)

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

'''
print('Renaming training images...')
for i in range(0, numberOfSamples):
    if i<10:
        os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/000' + str(i) + '.png')
        os.rename(datalocationpath + "mask/segmentation_" + str(i + 2) + ".png", datalocationpath + 'mask/000' + str(i) + '.png')
    elif i<100:
        os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/00' + str(i) + '.png')
        os.rename(datalocationpath + "mask/segmentation_" + str(i + 2) + ".png", datalocationpath + 'mask/00' + str(i) + '.png')
    elif i<1000:
        os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/0' + str(i) + '.png')
        os.rename(datalocationpath + "mask/segmentation_" + str(i + 2) + ".png", datalocationpath + 'mask/0' + str(i) + '.png')     
    else:
        os.rename(datalocationpath + "rgb/rgb_" + str(i + 2) + ".png", datalocationpath + 'rgb/' + str(i) + '.png')
        os.rename(datalocationpath + "mask/segmentation_" + str(i + 2) + ".png", datalocationpath + 'mask/' + str(i) + '.png')
'''
print("Done!")
