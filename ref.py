import os
import numpy as np
from glob import glob
import cv2
from skimage.measure import compare_mse, compare_ssim, compare_psnr
import argparse

def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def mse(tf_img1, tf_img2):
    return compare_mse(tf_img1, tf_img2)


def psnr(tf_img1, tf_img2):
    return compare_psnr(tf_img1, tf_img2)


def ssim(tf_img1, tf_img2):
    return compare_ssim(tf_img1, tf_img2)


def main(args):

    #WSI_MASK_PATH1 = 'C:/Users/Lebron/Desktop/result_images/Clean_Images/CityScapecut/'
    #WSI_MASK_PATH2 = 'C:/Users/Lebron/Desktop/result_images/result_Zero_DCE++4/CityScape/'

    WSI_MASK_PATH1 = args.test_dir1
    print("path 1 is", WSI_MASK_PATH1)
    WSI_MASK_PATH2 = args.test_dir2
    print("path 2 is", WSI_MASK_PATH2)

    path_real = glob(os.path.join(WSI_MASK_PATH1, '*.png'))
    path_fake = glob(os.path.join(WSI_MASK_PATH2, '*.png'))

    list_psnr = []
    list_ssim = []
    list_mse = []

    for i in range(len(path_real)):
        t1 = read_img(path_real[i])
        t2 = read_img(path_fake[i])
        result1 = np.zeros(t1.shape, dtype=np.float32)
        result2 = np.zeros(t2.shape, dtype=np.float32)
        cv2.normalize(t1, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.normalize(t2, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mse_num = mse(result1, result2)
        psnr_num = psnr(result1, result2)
        ssim_num = ssim(result1, result2)
        list_psnr.append(psnr_num)
        list_ssim.append(ssim_num)
        list_mse.append(mse_num)

        # Score for each image
        #print("{}/".format(i + 1) + "{}:".format(len(path_real)))
        #str = "\\"
        #print("image:" + path_real[i][(path_real[i].index(str) + 1):])
        #print("PSNR:", psnr_num)
        #print("SSIM:", ssim_num)
        #print("MSE:", mse_num)

    # Average score for the dataset
    print("Average PSNR:", "%.3f" % (np.mean(list_psnr)))
    print("Average SSIM:", "%.3f" % (np.mean(list_ssim)))
    print("Average MSE:", "%.3f" % (np.mean(list_mse)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dir1', type=str,
                        default='C:/Users/Lebron/Desktop/result_images/Clean_Images/CityScapecut/',
                        help='directory for clean images')
    parser.add_argument('--test_dir2', type=str,
                        default='C:/Users/Lebron/Desktop/result_images/result_Zero_DCE++4/CityScape/',
                        help='directory for enhanced or restored images')
    args = parser.parse_args()
    main(args)
