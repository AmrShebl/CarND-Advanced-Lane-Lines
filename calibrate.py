import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np



def calibrate_camera(dir, n_corners):
    images = glob.glob(dir)
    obj_points=[]
    img_points=[]
    obj_p = np.zeros((n_corners[0] * n_corners[1], 3), np.float32)
    obj_p[:,:2]= np.mgrid[0:n_corners[0],0:n_corners[1]].T.reshape(-1,2)
    img_size=0
    for imagef in images:
        img = cv2.imread(imagef)
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_size=gray_img.shape[::-1]
        ret, img_p=cv2.findChessboardCorners(gray_img,n_corners)
        if ret:
            obj_points.append(obj_p)
            img_points.append(img_p)
    ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(obj_points,img_points,img_size,None,None)

    return mtx, dist