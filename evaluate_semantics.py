import semantics as sem
import cv2
import numpy as np

def main():
    boardstates_path = "datasets/ButtonPose/data/boardstates.json"
    capturedata_path = "capturedata.json"
    model_dict_path = "./models/ply"

    AD_savepath = "./AD_confusions.csv"
    CD_savepath = "./CD_confusions.csv"

    buttons = ["redbutton", "arrowbutton", "mushroombutton"]
    boards = {"2-slot": 2, "3-slot": 3}
    class_to_name = {1 : "2-slot", 2 : "3-slot", 3 : "mushroombutton", 4 : "arrowbutton", 5 : "redbutton", 6 : "unknownbutton"}
    name_to_model = sem.load_3D_models(class_to_name, model_dict_path)

    translations_to_slots = sem.get_hole_translations()
    boardstates, capturedata, _ = sem.load_data(boardstates_path, capturedata_path)

    threshold = 40
    maximum_threshold = 80
    threshold_interval = 0.5

    start_index = 18000
    stop_index = 20000

    AD_data = []
    CD_data = []

    while threshold < maximum_threshold:
        print("Calculating for threshold: " + str(threshold) + " mm...")
        Cmatrix_AD, Cmatrix_CD = evaluate_threshold(start_index, stop_index, capturedata, boardstates, buttons, boards, name_to_model, translations_to_slots, threshold)
        AD_data.append([threshold, Cmatrix_AD])
        CD_data.append([threshold, Cmatrix_CD])
        threshold += threshold_interval

    print("Saving data...")
    save_to_csv(AD_data, AD_savepath)
    save_to_csv(CD_data, CD_savepath)
    print("Done.")


def evaluate_threshold(start_index, stop_index, capturedata, boardstates, buttons, boards, name_to_model, translations_to_slots, threshold):
    Cmatrix_AD = ConfusionMatrix()
    Cmatrix_CD = ConfusionMatrix()

    for i in range(start_index, stop_index):
        print(str(i-start_index) + "/" + str(stop_index-start_index), end='\r')
        capture = capturedata[str(i)]
        boardstate = boardstates[str(i)]

        estimated_boardstate_AD = estimate_boardstate_AD(capture, buttons, boards, translations_to_slots, name_to_model, threshold)
        estimated_boardstate_CD = estimate_boardstate_CD(capture, buttons, boards, translations_to_slots, threshold)

        Cmatrix_AD.add(evaluate_boardstate(estimated_boardstate_AD, boardstate))
        Cmatrix_CD.add(evaluate_boardstate(estimated_boardstate_CD, boardstate))

    return Cmatrix_AD, Cmatrix_CD

def estimate_boardstate_AD(capture, buttons, boards, translations_to_slots, name_to_model, threshold):
    boardstate = generate_empty_boardstate(boards)

    for board in [board for board in boards.keys() if board in capture["translations"].keys()]:
        boardposition = capture["translations"][board]
        boardrotation, _ = cv2.Rodrigues(np.array(capture["rotations"][board], dtype=np.float32))
            
        for button in [button for button in buttons if button in capture["translations"].keys()]:
            buttonposition = capture["translations"][button]
            buttonrotation, _ = cv2.Rodrigues(np.array(capture["rotations"][button], dtype=np.float32))

            for i in range(boards[board]):
                if boardstate[board][i] != 'empty':
                    continue

                holeposition = np.matmul(boardrotation, translations_to_slots[button][board][i]) + boardposition
                holerotation = np.matmul(boardrotation, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

                average_distance = sem.compute_AD(name_to_model[button], holerotation, holeposition, buttonrotation, buttonposition)
                
                if average_distance < threshold:
                    boardstate[board][i] = button
                    break
    
    return boardstate

def estimate_boardstate_CD(capture, buttons, boards, translations_to_slots, threshold):
    boardstate = generate_empty_boardstate(boards)

    for board in [board for board in boards.keys() if board in capture["translations"].keys()]:
        boardposition = capture["translations"][board]
        boardrotation, _ = cv2.Rodrigues(np.array(capture["rotations"][board], dtype=np.float32))
            
        for button in [button for button in buttons if button in capture["translations"].keys()]:
            buttonposition = capture["translations"][button]

            for i in range(boards[board]):
                if boardstate[board][i] != 'empty':
                    continue

                holeposition = np.matmul(boardrotation, translations_to_slots[button][board][i]) + boardposition

                centerdistance = np.linalg.norm(buttonposition - holeposition)
                
                if centerdistance < threshold:
                    boardstate[board][i] = button
                    break
    
    return boardstate

def evaluate_boardstate(estimated_boardstate, gt_boardstate):
    Cmatrix = ConfusionMatrix()
    for board in gt_boardstate.keys():
        if board not in estimated_boardstate:
            continue

        gt = gt_boardstate[board]
        estimated = estimated_boardstate[board]

        Cmatrix_normal = evaluate_board(estimated, gt)
        Cmatrix_reversed =  evaluate_board(estimated[::-1], gt)

        if Cmatrix_normal.TP + Cmatrix_normal.TN >= Cmatrix_reversed.TP +  Cmatrix_reversed.TN:
            Cmatrix.add(Cmatrix_normal)
        else:
            Cmatrix.add(Cmatrix_reversed)
    
    return Cmatrix

        

def evaluate_board(estimated_state, gt_state):
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

def generate_empty_boardstate(boards):
    boardstate = {}
    for board in boards:
        emptyarray = ["empty" for i in range(boards[board])]
        boardstate[board] = emptyarray

    return boardstate


def save_to_csv(data, filepath):
    with open(filepath, "a") as csvfile:
        for frame in data:
            threshold = frame[0]
            Cmatrix = frame[1]

            csvfile.write(str(threshold) + ',' + str(Cmatrix.TP) + ',' + str(Cmatrix.FP) + ',' + str(Cmatrix.FN) + ',' + str(Cmatrix.TN) + '\n')

        

class ConfusionMatrix:
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