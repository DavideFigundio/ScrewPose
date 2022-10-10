import cv2
from os import listdir
from os.path import join

def main():
    dirpath = "../background_extra/"
    newsize = (1280, 720)

    images = [f for f in listdir(dirpath) if f.endswith(".png")]
    for imgname in images:
        print(imgname)
        image = cv2.imread(join(dirpath, imgname))
        cv2.imwrite(join(dirpath, imgname), cv2.resize(image, newsize))

if __name__ == "__main__":
    main()
