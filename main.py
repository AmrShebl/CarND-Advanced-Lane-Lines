from calibrate import calibrate_camera
from pipeline import process
import cv2
import glob
import matplotlib.pyplot as plt

def main():
    dir = './camera_cal/calibration*.jpg'
    mtx, dist = calibrate_camera(dir,(9,6))
    dir = './test_images/test*.jpg'
    #dir = './test_images/straight_lines*.jpg'
    imageFs=glob.glob(dir)
    for imageF in imageFs:
        img = cv2.imread(imageF)
        process(img, mtx, dist)



    # images = glob.glob(dir)
    # for imagef in images:
    #     img = cv2.imread(imagef)
    #     undist_img=cv2.undistort(img,mtx,dist,None,mtx)
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #     undist_img = cv2.cvtColor(undist_img,cv2.COLOR_BGR2RGB)
    #     f, (x1,x2)=plt.subplots(1,2)
    #     x1.imshow(img)
    #     x1.set_title('Original Image')
    #     x2.imshow(undist_img)
    #     x2.set_title('Undistorted Image')
    #     #plt.imshow(img)
    # plt.show()

if __name__ == "__main__":
    main()