import numpy as np
import json
import distributions
import cv2
import yaml

def main():
    boardstates_path = "datasets/ButtonPose/data/boardstates.json"
    capturedata_path = "capturedata.json"
    gt_path = "datasets/ButtonPose/data/gt.yml"

    buttons = ["redbutton", "arrowbutton", "mushroombutton"]
    boardsizes = {"2-slot": 2, "3-slot": 3}
    class_to_name = {1 : "2-slot", 2 : "3-slot", 3 : "mushroombutton", 4 : "arrowbutton", 5 : "redbutton", 6 : "unknownbutton"}

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
                        distance = np.linalg.norm(buttonposition - holeposition)

                        distribution_graph.insert_value(distance, boardstate[board][j] == button)
    
    distribution_graph.save_csv("distribution.csv")

def load_data(boardstates_path, capturedata_path, gt_path):
    with open(capturedata_path, "r") as file:
        capturedata = json.load(file)

    with open(boardstates_path) as file:
        boardstates = json.load(file)

    with open(gt_path, "r") as file:
        ground_truths = yaml.safe_load(file)

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

if __name__ == "__main__":
    main()