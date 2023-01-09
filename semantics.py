import numpy as np
import json
import distributions
import os
from plyfile import PlyData
import cv2
from utils.compute_overlap import wrapper_c_min_distances
import yaml

def main():
    boardstates_path = "datasets/ButtonPose/data/boardstates.json"
    capturedata_path = "capturedata.json"
    gt_path = "datasets/ButtonPose/data/gt.yml"
    model_dict_path = "./models/ply"

    buttons = ["redbutton", "arrowbutton", "mushroombutton"]
    boardsizes = {"2-slot": 2, "3-slot": 3}
    class_to_name = {1 : "2-slot", 2 : "3-slot", 3 : "mushroombutton", 4 : "arrowbutton", 5 : "redbutton", 6 : "unknownbutton"}
    name_to_model = load_3D_models(class_to_name, model_dict_path)

    translations_to_slots = get_hole_translations()
    boardstates, capturedata, ground_truths = load_data(boardstates_path, capturedata_path, gt_path)
    
    minimum_mm = 0
    maximum_mm = 50
    interval_mm = 1
    distribution_graph = distributions.BinaryDistributionGraph(minimum=minimum_mm, maximum=maximum_mm, interval=interval_mm)

    for i in range(18000, 20000):
        capture = capturedata[str(i)]
        boardstate = boardstates[str(i)]
        gt = ground_truths[i]

        for button in buttons:
            buttonposition = capture["translations"][button]

            if buttonposition == [0, 0, 0]:
                continue

            for board in boardsizes.keys():
                board_rotation_matrix = np.array([data["cam_R_m2c"] for data in gt if class_to_name[int(data["obj_id"])] == board], dtype=np.float32)

                if board_rotation_matrix.size > 0:
                    board_rotation_matrix = board_rotation_matrix.reshape(3, 3)
                    board_position = np.array([data["cam_t_m2c"] for data in gt if class_to_name[int(data["obj_id"])] == board], dtype=np.float32)

                    for j in range(boardsizes[board]):
                        holeposition = np.matmul(board_rotation_matrix, translations_to_slots[button][board][j]) + board_position
                        holerotation = np.matmul(board_rotation_matrix, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

                        buttonrotation, _ = cv2.Rodrigues(np.array(capture["rotations"][button], dtype=np.float32))

                        # distance = np.linalg.norm(buttonposition - holeposition)
                        distance = compute_AD(name_to_model[button], holerotation, holeposition, buttonrotation, buttonposition)

                        distribution_graph.insert_value(distance, boardstate[board][j] == button)
    
    distribution_graph.save_csv("distribution_AD.csv")

def compute_AD(modelpoints, gt_rot_mat, gt_trans, pred_rot_mat, pred_trans, maxpoints = 1000):

    transformed_points_gt = np.matmul(modelpoints, gt_rot_mat.T) + gt_trans
    transformed_points_pred = np.matmul(modelpoints, pred_rot_mat.T) + pred_trans

    num_points = transformed_points_gt.shape[0]
    
    #approximate the add-s metric and use max max_points of the 3d model points to reduce computational time
    step = num_points // maxpoints + 1
    
    min_distances = wrapper_c_min_distances(transformed_points_gt[::step, :], transformed_points_pred[::step, :])
    return np.mean(min_distances)

def load_data(boardstates_path, capturedata_path, gt_path=None):

    print("Loading inferences...")
    with open(capturedata_path, "r") as file:
        capturedata = json.load(file)

    print("Loading boardstates...")
    with open(boardstates_path) as file:
        boardstates = json.load(file)

    if gt_path != None:
        print("Loading ground truths...")
        with open(gt_path, "r") as file:
            ground_truths = yaml.safe_load(file)
    else:
        ground_truths = None

    print("Done!")
    return boardstates, capturedata, ground_truths

def get_hole_translations():
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
        
        name_to_model_dict[class_to_name_dict[key]] = points

    return name_to_model_dict
