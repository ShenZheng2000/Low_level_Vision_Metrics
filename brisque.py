import os
from PIL import Image
import numpy as np
from skimage import io, img_as_float
# pip install image-quality
import imquality.brisque as brisque
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dir', type=str,
                        help='directory for testing inputs')
    args = parser.parse_args()

    path = args.test_dir
    pathList = os.listdir(path)

    total = 0
    count = 0

    print("start calculating brisque")
    for item in pathList:
        print(item)

        # img = np.array(Image.open(os.path.join('F:/test1/', item)).convert('LA'))[:, :, 0]
        imgOri = Image.open(os.path.join(path, item))
        # img = np.array(imgOri.convert('LA'))[:, :, 0]
        img = img_as_float(imgOri)
        score = brisque.score(img)

        total += score
        count += 1

    print("The average BRISQUE score is", "%.3f" % (total / count))


