'''
Script for evaluating the performance of two different semantic thresholding methods.
These methods differ based on whether they use the Average Symmetric Distance (AD) or
the Center Distance (CD) to compare with the threshold.

This script requires previously obtained data, obtained from generation and inferencing of the
ButtonPose dataset:
    -the ground truth semantic state of each inference, contained in boardstates.json
    -the result of inferencing on the dataset, contained in capturedata.json. Alternatively,
     this script could be modified to directly take the output of the neural network, if the
     ground truth is known and constant.
    -the 3D models of the dataset objects to perform the Average Symmetric Distance computation,
     in .ply format.

This script saves the confusion matrices for all the specified thresholds, outputting two .csv 
files, one for each method. The files are saved in the format:

    Column 1:       |Column 2:      |Column 3:      |Column 4:      |Column 5:
    Threshold [mm]  |True Positives |False Positives|False Negatives|True Negatives
'''

import cv2
import numpy as np
import json
import os
from plyfile import PlyData
from utils.compute_overlap import wrapper_c_min_distances

def main():
    boardstates_path = "datasets/ButtonPose/data/boardstates.json" # Path to the boardstates.json file
    capturedata_path = "./data/capturedata.json"                   # Path to the capturedata.json file
    model_path = "./models/ply"                                    # Path to the directory containing the 3D models

    # Save paths for the output .csv files
    AD_savepath = "./data/AD_confusions.csv"
    CD_savepath = "./data/CD_confusions.csv"

    # Information on the dataset objects
    buttons = ["redbutton", "arrowbutton", "mushroombutton"]    # List of the names of button objects
    boards = {"2-slot": 2, "3-slot": 3}                         # Dict that associates each board name with the number of slots it has
    class_to_name = {1 : "2-slot", 2 : "3-slot", 3 : "mushroombutton", 4 : "arrowbutton", 5 : "redbutton", 6 : "unknownbutton"}

    # Loading 3D models
    name_to_model = load_3D_models(class_to_name, model_path)

    # Getting the position of each hole for each board
    translations_to_slots = get_hole_translations()

    # Loading data
    boardstates, capturedata = load_data(boardstates_path, capturedata_path)

    # Specifications for thresholds. All values expressed in mm.
    threshold = 2
    maximum_threshold = 150
    threshold_interval = 2

    # Indexes that indicate the captures to be considered from the ButtonPose dataset.
    # Images 18000-20000 are the same used for evaluation during training.
    start_index = 18000
    stop_index = 20000

    # Lists used to save data for each iteration.
    AD_data = []
    CD_data = []

    # Iterating over defined thresholds.
    while threshold < maximum_threshold:
        print("Calculating for threshold: " + str(threshold) + " mm...")

        # Computation of the confusion matrices for each threshold.
        Cmatrix_AD, Cmatrix_CD = evaluate_threshold(start_index, stop_index, capturedata, boardstates, buttons, boards, name_to_model, translations_to_slots, threshold)

        AD_data.append([threshold, Cmatrix_AD])
        CD_data.append([threshold, Cmatrix_CD])

        threshold += threshold_interval

    # Saving data to .csv format once computation has concluded.
    print("Saving data...")
    save_to_csv(AD_data, AD_savepath)
    save_to_csv(CD_data, CD_savepath)
    print("Done.")


def evaluate_threshold(start_index, stop_index, capturedata, boardstates, buttons, boards, name_to_model, translations_to_slots, threshold):
    '''
    Evaluates the perfromance of the Average Distance and Center Distance Methods at a certain threshold.
    Arguments:
        -start_index, stop_index - integers representing the indexes that delimitate the frames that are considered in the evalutation
        -capturedata - object containing inference data for all frames. See capturedata.json for structure.
        capturedata -> (frame number [str]) -> ("rotation"/"translation"[str]) -> (object name[str]) -> (data [int array])
        -boardstates - object containing ground truth semantic data for all frames. See boardstates.json for structure
        boardstates -> (frame number [str]) -> (board name[str]) -> (slots [str array])
        Each slot is either "empty" or contains the name of the button that fills it.
        -buttons - list of names of the buttons to be placed
        -boards - dict that associates each board name (str) with the number of slots it has(int).
        -name_to_model - dict that associates each object name with a 3xn array containing the points of its 3D model.
        -translations_to_slots - dict that associates each board with the positions from its center to the center of its slots.
        These positions vary based on the button considered, as they may have different shapes.
        -threshold - int [mm] the threshold for evaluating position.
    Returns
        -Cmatrix_AD - confusion matrix object for the Average Distance method.
        -Cmatrix_CD - confusion matrix object for the Center Distance method.
    '''
    Cmatrix_AD = ConfusionMatrix()
    Cmatrix_CD = ConfusionMatrix()

    # Iterating over the defined indexes
    for i in range(start_index, stop_index):
        print(str(i-start_index) + "/" + str(stop_index-start_index), end='\r')

        # Extracting capture data and semantic data related to each frame
        capture = capturedata[str(i)]
        boardstate = boardstates[str(i)]

        # Estimating the state of the capture using the two different methods
        estimated_boardstate_AD = estimate_boardstate_AD(capture, buttons, boards, translations_to_slots, name_to_model, threshold)
        estimated_boardstate_CD = estimate_boardstate_CD(capture, buttons, boards, translations_to_slots, threshold)

        # Evaluating and adding the confusion matrices to the previous results.
        Cmatrix_AD.add(evaluate_boardstate_symmetrical(estimated_boardstate_AD, boardstate))
        Cmatrix_CD.add(evaluate_boardstate_symmetrical(estimated_boardstate_CD, boardstate))

    return Cmatrix_AD, Cmatrix_CD

def estimate_boardstate_AD(capture, buttons, boards, translations_to_slots, name_to_model, threshold):
    '''
    Estimates the semantic state of a frame using the Average Distance method.
    Uses a two-step approach, first computing all objects that fit under the threshold,
    and then resolving conflicts based on the shortest distance.
    Arguments:
        -capture - inference data from a single frame, containing all rotations and translations for all objects present.
        (capture) -> ("rotation"/"translation"[str]) -> (object name[str]) -> (data [int array])
        -buttons - list of names of the buttons to be placed
        -boards - dict that associates each board name (str) with the number of slots it has(int).
        -name_to_model - dict that associates each object name with a 3xn array containing the points of its 3D model.
        -translations_to_slots - dict that associates each board with the positions from its center to the center of its slots.
        These positions vary based on the button considered, as they may have different shapes.
        -threshold - int [mm] the threshold for evaluating position.
    Returns:
        -boardstate - object that represents the estimated state of the slots on each board for the frame.
        boardstate -> (board name[str]) -> (slots [str array])
    '''

    # Generates a dict that associates each slot to a dict of plausible buttons and their distances from the slot
    boardstate = generate_empty_boardstate_dict(boards)

    # Iterating over boards detected in the capture
    for board in [board for board in boards if board in capture["translations"]]:
        # Extracting position and rotation of the board
        boardposition = capture["translations"][board]
        boardrotation, _ = cv2.Rodrigues(np.array(capture["rotations"][board], dtype=np.float32))
        
        # Iterating over buttons detected in the capture
        for button in [button for button in buttons if button in capture["translations"]]:
            # Extracting position and rotation of the button
            buttonposition = capture["translations"][button]
            buttonrotation, _ = cv2.Rodrigues(np.array(capture["rotations"][button], dtype=np.float32))

            # Iterating over slots for each board.
            for i in range(boards[board]):
                # For each slot, compute position and orientation of a button if it were inside the slot
                holeposition = np.matmul(boardrotation, translations_to_slots[button][board][i]) + boardposition
                # Rotation matrix from board orientation to slot orientation is a constant -90 degree rotation
                # around the y axis.
                holerotation = np.matmul(boardrotation, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

                # Computing the average distance between the estimated button position and the position of the button
                # if it were in the estimated position of the slot.
                average_distance = compute_AD(name_to_model[button], holerotation, holeposition, buttonrotation, buttonposition)
                
                # If the average distance is less than the threshold, it is added to the possible options for the slot.
                if average_distance < threshold:
                    boardstate[board][i][button] = average_distance

    # Evaluation of each slot to determine final estimation.
    return parse_board_estimations(buttons, boards, boardstate)


def estimate_boardstate_CD(capture, buttons, boards, translations_to_slots, threshold):
    '''
    Estimates the semantic state of a frame using the Center Distance method.
    Uses a two-step approach, first computing all objects that fit under the threshold,
    and then resolving conflicts based on the shortest distance.
    Arguments:
        -capture - inference data from a single frame, containing all rotations and translations for all objects present.
        (capture) -> ("rotation"/"translation"[str]) -> (object name[str]) -> (data [int array])
        -buttons - list of names of the buttons to be placed
        -boards - dict that associates each board name (str) with the number of slots it has(int).
        -translations_to_slots - dict that associates each board with the positions from its center to the center of its slots.
        These positions vary based on the button considered, as they may have different shapes.
        -threshold - int [mm] the threshold for evaluating position.
    Returns:
        -boardstate - object that represents the estimated state of the slots on each board for the frame.
        boardstate -> (board name[str]) -> (slots [str array])
    '''
    # Generates a dict that associates each slot to a dict of plausible buttons and their distances from the slot
    boardstate = generate_empty_boardstate_dict(boards)

    # Iterating over boards detected in the capture
    for board in [board for board in boards if board in capture["translations"]]:
        # Extracting position of the board
        boardposition = capture["translations"][board]
        boardrotation, _ = cv2.Rodrigues(np.array(capture["rotations"][board], dtype=np.float32))
        
        # Iterating over buttons detected in the capture
        for button in [button for button in buttons if button in capture["translations"]]:
            # Extracting position of the button
            buttonposition = capture["translations"][button]

            # Iterating over slots for each board.
            for i in range(boards[board]):
                # For each slot, compute position of a button if it were inside the slot
                holeposition = np.matmul(boardrotation, translations_to_slots[button][board][i]) + boardposition

                # Computing the distance between the estimated button position and the position of the button
                # if it were in the estimated position of the slot.
                centerdistance = np.linalg.norm(buttonposition - holeposition)
                
                # If the average distance is less than the threshold, it is added to the possible options for the slot.
                if centerdistance < threshold:
                    boardstate[board][i][button] = centerdistance

    # Evaluation of each slot to determine final estimation.
    return parse_board_estimations(buttons, boards, boardstate)

def parse_board_estimations(buttons, boards, boardstate):
    '''
    Resolves conflicts between plausible placements based on the shortest distance.
    Arguments:
        -buttons - list of names of the buttons to be placed
        -boards - dict that associates each board name (str) with the number of slots it has(int).
        -boardstate - object that associates each board with its slots, each slot with a list of possible 
        candidate buttons, and each candidate with its distance from the slot
    Returns:
        -boardstate - object that represents the estimated state of the slots on each board for the frame.
        boardstate -> (board name[str]) -> (slots [str array])
    '''
    final_boardstate = generate_empty_boardstate(boards)

    # Iterating over all buttons
    for button in buttons:
        placed = False  # Represents if a button has been placed in a slot definitively or not

        # Generation of a dict that for the selected button, associates each distance from a plausible placement
        # to the board and slot of the placement.
        distances = {boardstate[board][i][button]:(board, i) for board in boardstate for i in range(boards[board]) if button in boardstate[board][i]}
        
        # If the button has not been identified by the network, distances will be empty.
        while not placed and distances:

            # Extracting the closest slot and board as a candidate.
            minimumdistance = min(distances.keys())
            closest_board = distances[minimumdistance][0]
            closest_slot = distances[minimumdistance][1]
            
            # If the candidate slot is the last possible placement, or the button in question is the closest to 
            # to the candidate slot, then the button is considered to be in that slot.
            if len(distances) == 1 or min(boardstate[closest_board][closest_slot].values()) == minimumdistance:
                final_boardstate[closest_board][closest_slot] = button
                placed = True
            
            # Otherwise we remove the candidate from the considered distances and continue to the next ones.
            else:
                distances.pop(minimumdistance)

    return final_boardstate

def evaluate_boardstate_symmetrical(estimated_boardstate, gt_boardstate):
    '''
    Evaluates an estimated boardstate against a ground truth boardstate.
    Since boards are symmetrical, both the normal and inverse orientation are considered,
    and the one with the best performance is considered to be correct.
    Arguments:
        -estimated_boardstate, gt_boardstate - object that represents the estimated and ground truth state of the slots on each board for the frame.
            boardstate -> (board name[str]) -> (slots [str array])
    
    Returns:
        -Cmatrix - a confusion matrix object containing evaluation data for the frame: True/False Positives/Negatives
    '''
    Cmatrix = ConfusionMatrix()

    # Evaluation is performed separately for each board
    for board in gt_boardstate.keys():
        if board not in estimated_boardstate:
            continue

        gt = gt_boardstate[board]
        estimated = estimated_boardstate[board]

        # Both normal and inverted conditions are considered
        Cmatrix_normal = evaluate_board(estimated, gt)
        Cmatrix_reversed =  evaluate_board(estimated[::-1], gt)

        # The best option is saved. "best" is considered to be the one with more positives.
        if Cmatrix_normal.TP + Cmatrix_normal.TN >= Cmatrix_reversed.TP +  Cmatrix_reversed.TN:
            Cmatrix.add(Cmatrix_normal)
        else:
            Cmatrix.add(Cmatrix_reversed)
    
    return Cmatrix

        

def evaluate_board(estimated_state, gt_state):
    '''
    Evaluates the condition of a single board against a ground truth.
    Arguments:
        -estimated_state, gt_state - lists that represent the estimated and ground truth state of the slots on the board.
            state -> (slots [str array])
    Returns:
        -Cmatrix - a confusion matrix object containing evaluation data for the board: True/False Positives/Negatives
    '''
    Cmatrix = ConfusionMatrix()
    for i in range(len(estimated_state)):
        if estimated_state[i] == gt_state[i]:
            if estimated_state[i] == 'empty':
                Cmatrix.TN += 1
            else:
                Cmatrix.TP += 1
        
        else:
            if estimated_state[i] == 'empty':
                Cmatrix.FN += 1
            else:
                Cmatrix.FP += 1
    
    return Cmatrix

def generate_empty_boardstate_dict(boards):
    '''
    Generates an empty boardstate object where each each board is associated with a list of slots, and each slot is represented
    by an empty dictionary.
    '''
    return {board:[{} for i in range(boards[board])] for board in boards}

def generate_empty_boardstate(boards):
    '''
    Generates an empty boardstate object where each each board is associated with a list of slots, and each slot is represented
    with a string.
    '''
    return {board:["empty" for i in range(boards[board])] for board in boards}



def save_to_csv(data, filepath):
    '''
    Saves evaluation data to a csv format. Caution! Overwrites previous data. If this is undesired, change open(filepath, "w")
    to open(filepath, "a") to set into append mode. The .csv file will be in the format:

        Threshold [mm]  |True Positives |False Positives|False Negatives|True Negatives

    Arguments:
        -data - a list of evalutaions for different thresholds. Each element is a tuple of the type (threshold, Cmatrix),
            where threshold is the considered threshold and Cmatrix is a confusion matrix object.
        -filepath - string indicating the path where the data is saved.
    '''
    with open(filepath, "w") as csvfile:
        for frame in data:
            threshold = frame[0]
            Cmatrix = frame[1]

            csvfile.write(str(threshold) + ',' + str(Cmatrix.TP) + ',' + str(Cmatrix.FP) + ',' + str(Cmatrix.FN) + ',' + str(Cmatrix.TN) + '\n')


def compute_AD(modelpoints, gt_rot_mat, gt_trans, pred_rot_mat, pred_trans, maxpoints = 1000):
    '''
    Computes the Symmetric Average Distance between a ground truth and a predicted pose.
    Arguments: 
        -modelpoints - 3xn numpy array containing 3D points for the model
        -gt_rot_mat - 3x3 numpy array containing ground truth rotation
        -gt_trans - 1x3 numpy array containing ground truth translation
        -pred_rot_mat - 3x3 numpy array containing estimated rotation
        -pred_trans - 1x3 numpy array containing estimated translation
        -maxpoints - maximum number of points considered in computation. Defaults to 1000.
    Returns:
        -The average symmetric distance in mm.
    '''

    # Generation of points for ground truth and prediction poses
    transformed_points_gt = np.matmul(modelpoints, gt_rot_mat.T) + gt_trans
    transformed_points_pred = np.matmul(modelpoints, pred_rot_mat.T) + pred_trans

    num_points = transformed_points_gt.shape[0]
    
    # Approximate the add-s metric and use max max_points of the 3d model points to reduce computational time
    step = num_points // maxpoints + 1
    
    # Sampling points from the model based on maxpoints and computing minimum distances
    min_distances = wrapper_c_min_distances(transformed_points_gt[::step, :], transformed_points_pred[::step, :])

    # Averaging minimum distances.
    return np.mean(min_distances)

def load_data(boardstates_path, capturedata_path):
    '''
    Loads the capturedata and boardstates from file.
    Arguments:
        -capturedata - path to capture data
        -boardstates - path to boardstates

    Returns:
        -capturedata, boardstates - objects built depending on the .json files.
    '''

    print("Loading inferences...")
    with open(capturedata_path, "r") as file:
        capturedata = json.load(file)

    print("Loading boardstates...")
    with open(boardstates_path) as file:
        boardstates = json.load(file)

    print("Done!")
    return boardstates, capturedata

def get_hole_translations():
    '''
    Generates a dict that associates for each button and for each slot the translations from the center of each board to the center
    of the button if it were in that slot.
    '''
    translation_to_slots = {
            "mushroombutton": {
                "2-slot": [
                    np.array([0, -15.06, 35.6]),
                    np.array([0, 15.06, 35.6])],
                "3-slot": [
                    np.array([0, -30.1, 35.6]),
                    np.array([0, 0, 35.6]),
                    np.array([0, 30.1, 35.6])]},
            "redbutton": {
                "2-slot": [
                    np.array([0, -15.06, 25.1]),
                    np.array([0, 15.06, 25.1])],
                "3-slot": [
                    np.array([0, -30.1, 25.1]),
                    np.array([0, 0, 25.1]),
                    np.array([0, 30.1, 25.1])]},
            "arrowbutton": {
                "2-slot": [
                    np.array([0, -15.06, 25.1]),
                    np.array([0, 15.06, 25.1])],
                "3-slot": [
                    np.array([0, -30.1, 25.1]),
                    np.array([0, 0, 25.1]),
                    np.array([0, 30.1, 25.1])]}}
    
    return translation_to_slots

def load_3D_models(class_to_name_dict, path_to_models):
    """
    Function that loads the 3d models from a directory. Models must be in the .ply format and each model must be named in the format "<name>.ply",
    where <name> is the name it has inside the class_to_name dictionary.
    Argumentss:
        class_to_name_dict - dictionary that associates each class to the corresponding model name
        path_to_models - location where the .ply files are stored
    Returns:
        name_to_model_dict - dictionary that associates each name to a 3xn array containing the 3D points of its model.
    """

    name_to_model_dict = {}

    for key in class_to_name_dict.keys():
        model_data = PlyData.read(os.path.join(path_to_models, class_to_name_dict[key] + ".ply"))
                                  
        vertex = model_data['vertex']
        points = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis = -1).astype(np.float32)
        
        name_to_model_dict[class_to_name_dict[key]] = points

    return name_to_model_dict

class ConfusionMatrix:
    '''
    Class used to represent a confusion matrix.
    Attribtes:
        -TP - Number of True Positives
        -FP - Number of False Positives
        -FN - Number of False Negatives
        -TN - Number of True Negatives
    Methods:
        -precision - returns the calculated precision
        -recall - returns the calculated recall
        -add - sums the matrix with the argument if it is also a ConfusionMatrix.
    '''
    def __init__(self, TP=0, TN=0, FP=0, FN=0):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

    def precision(self):
        if self.TP + self.FP > 0:
            return self.TP/(self.TP + self.FP)
        else:
            return None

    def recall(self):
        if self.TP + self.FN > 0:
            return self.TP/(self.TP + self.FP)
        else:
            return None
        
    def add(self, Cmatrix):
        if type(self) != type(Cmatrix):
            raise TypeError("Argument must be another Confusion Matrix")
        else:
            self.TP += Cmatrix.TP
            self.TN += Cmatrix.TN
            self.FP += Cmatrix.FP
            self.FN += Cmatrix.FN

    def __str__(self):
        return "[TP: " + str(self.TP) + ", FP: " + str(self.FP) + ", FN: " + str(self.FN) + ", TN: " + str(self.TN) + "]"

if __name__ == "__main__":
    main()