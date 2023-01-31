import cv2
import numpy as np

class BoardState:

    def __init__(self, boardsizes, buttonnames, translations_to_slots, boardstate=None):

        self.boardsizes = boardsizes
        self.buttonnames = buttonnames
        self.translations_to_slots = translations_to_slots

        if boardstate:
            self.boards = boardstate
        else:
            self.boards = {board:["" for i in range(boardsizes[board])] for board in boardsizes}
        
    def estimate(self, class_to_name, labels, rotations, translations, threshold) -> None:
        labels = [class_to_name[label] for label in labels]
        boardstate_estimations = {boardname:[{} for i in range(self.boardsizes[boardname])] for boardname in self.boardsizes.keys() if boardname in labels}

        for boardname in boardstate_estimations:
            boardindex = labels.index(boardname)
            boardposition = translations[boardindex]
            boardrotation, _ = cv2.Rodrigues(rotations[boardindex])

            for buttonname in [buttonname for buttonname in self.buttonnames if buttonname in labels]:
                buttonindex = labels.index(buttonname)
                buttonposition = translations[buttonindex]

                for i in range(self.boardsizes[boardname]):
                    # For each slot, compute position of a button if it were inside the slot
                    slotposition = np.matmul(boardrotation, self.translations_to_slots[buttonname][boardname][i]) + boardposition

                    # Computing the distance between the estimated button position and the position of the button
                    # if it were in the estimated position of the slot.
                    distance = np.linalg.norm(buttonposition - slotposition)
                    
                    # If the average distance is less than the threshold, it is added to the possible options for the slot.
                    if distance < threshold:
                        boardstate_estimations[boardname][i][buttonname] = distance
        
        # Iterating over all buttons
        for button in self.buttonnames:
            placed = False  # Represents if a button has been placed in a slot definitively or not

            # Generation of a dict that for the selected button, associates each distance from a plausible placement
            # to the board and slot of the placement.
            distances = {boardstate_estimations[board][i][button]:(board, i) for board in boardstate_estimations for i in range(self.boardsizes[board]) if button in boardstate_estimations[board][i]}
            
            # If the button has not been identified by the network, distances will be empty.
            while not placed and distances:

                # Extracting the closest slot and board as a candidate.
                minimumdistance = min(distances.keys())
                closest_board = distances[minimumdistance][0]
                closest_slot = distances[minimumdistance][1]
                
                # If the candidate slot is the last possible placement, or the button in question is the closest to 
                # to the candidate slot, then the button is considered to be in that slot.
                if len(distances) == 1 or min(boardstate_estimations[closest_board][closest_slot].values()) == minimumdistance:
                    self.boards[closest_board][closest_slot] = button
                    placed = True
                
                # Otherwise we remove the candidate from the considered distances and continue to the next ones.
                else:
                    distances.pop(minimumdistance)

    def clear(self) -> None:
        self.boards = {board:["" for i in range(self.boardsizes[board])] for board in self.boardsizes}

    def __str__(self) -> str:
        lines = ''
        for board in self.boards:
            lines += board + ": " + str(self.boards[board]) + "\n"

        return lines