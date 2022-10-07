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
    datalocationpath = './datasets/ScrewDataset/data/00/'
    numberOfSamples = 10000
    trainamount = 9000
    samplesPerCaptureFile = 150

    Cam_intrinsic = [640.0, 0., 640.0, 0., 640.0, 360.0, 0., 0., 1.]

    deletePreviousGT(datalocationpath)
    
    createLabels(datalocationpath, samplesPerCaptureFile, numberOfSamples, Cam_intrinsic)

    createSplits(datalocationpath, trainamount, numberOfSamples)

    renameImages(datalocationpath, numberOfSamples)

    print("Done!")

def createLabels(datalocationpath, samplesPerCaptureFile, numberOfSamples, Cam_intrinsic):
    totalCaptures = ceil(numberOfSamples/samplesPerCaptureFile)
    print("Creating ground truth labels...")
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

                    screw_present = False
                    assembly_present = False

                    for annotation in jsonData['captures'][i]['annotations']:

                        if annotation['id'] == 'bounding box':

                            for bboxdata in annotation['values']:
                                if bboxdata['label_name'] == "Bolt":
                                    screwbbox = [bboxdata['x'], bboxdata['y'], bboxdata['width'], bboxdata['height']]
                                    screw_present = True

                                elif bboxdata['label_name'] == "Assembly":
                                    assemblybbox = [bboxdata['x'], bboxdata['y'], bboxdata['width'], bboxdata['height']]
                                    assembly_present = True
                    
                        elif annotation['id'] == 'bounding box 3D':

                            for bboxdata in annotation['values']:
                                if bboxdata['label_name'] == "Bolt":
                                    translationdata = bboxdata['translation']
                                    screwtranslation = [1000*translationdata['x'], -1000*translationdata['y'], 1000*translationdata['z']]

                                    rotationdata = bboxdata['rotation']
                                    screwrotation = convertQuaternionToMatrix(rotationdata['x'], rotationdata['y'], rotationdata['z'], rotationdata['w'])
                                elif bboxdata['label_name'] == "Assembly":
                                    translationdata = bboxdata['translation']
                                    assemblytranslation = [1000*translationdata['x'], -1000*translationdata['y'], 1000*translationdata['z']]

                                    rotationdata = bboxdata['rotation']
                                    assemblyrotation = convertQuaternionToMatrix(rotationdata['x'], rotationdata['y'], rotationdata['z'], rotationdata['w'])    
                    
                    datalist = []

                    if screw_present:
                        datalist.append({'cam_R_m2c':screwrotation, 'cam_t_m2c':screwtranslation, 'obj_bb':screwbbox, 'obj_id':1})
                        with open(join(datalocationpath, "valid_poses/1.txt"), "a") as file:
                            file.write(str(j) + '\n')

                    if assembly_present:
                        datalist.append({'cam_R_m2c':assemblyrotation, 'cam_t_m2c':assemblytranslation, 'obj_bb':assemblybbox, 'obj_id':2})
                        with open(join(datalocationpath, "valid_poses/2.txt"), "a") as file:
                            file.write(str(j) + '\n')
        
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

if __name__ == "__main__":
    main()