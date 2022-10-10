from os import listdir, rename
from os.path import join

def main():
    dirpath = "../background/"

    j= 0
    images = [f for f in listdir(dirpath) if f.endswith(".png")]
    for imgname in images:
        newname = str(j) + ".png"
        rename(join(dirpath, imgname), join(dirpath, newname))
        j += 1

if __name__ == "__main__":
    main()