# Low_level_Vision_Metrics
This is a Python implementation of various low-level vision metrics. 

Up till now (2021.9.22), we support the implementation of reference metrics including PSN, SSIM and MSE, and no-reference metrics including UNIQUE and BRISQUE.

Note: For the reference metrics, all the predicted images and the groundtruth should be in `.png` format. However, you can change the code in `ref.py` to copy with images of other formats.

# Requirements
- Python3
- Opencv-python
- Scikit-image
- Numpy

# Quick Run
## Reference Metrics

To obtain the PSNR, SSIM and MSE score , run in terminal
```
python ref.py --test_dir1 path_to_pred_image --test_dir2 path_to_gt_image
```

For example, if your predicted image is at folder `/image/pred/`, and the corresponding groundtruth is at folder `/image/gt/`, you should   run
```
python ref.py --test_dir1 /image/pred/ --test_dir2 /image/gt/
```

## Non-reference Metrics

To obtain the BRISQUE score, run in terminal
```
python brisque.py -test_dir path_to_pred_image
```

The calculation of UNIQUE score is more complicated. 
Please go to this [website](https://github.com/zwx8981/UNIQUE) and download the repository and the weights according to the instructions.


# Related Repository
https://github.com/andrewekhalel/sewar

# Contact
Please contact zhengsh@kean.edu for any questions. You can also open an issue in this Github repository
