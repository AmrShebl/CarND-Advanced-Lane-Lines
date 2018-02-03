import cv2
import matplotlib.pyplot as plt
import numpy as np

def abs_sobel_threshol(img,orient='x',sobel_kernel=3,threshold=(0,255)):
    #gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img=img
    if orient=='x':
        sobel_img=cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    else:
        sobel_img=cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    sobel_img=np.abs(sobel_img)
    sobel_img=np.uint8(255*sobel_img/np.max(sobel_img))
    binary_img=np.zeros_like(sobel_img)
    binary_img[(sobel_img>threshold[0])&(sobel_img<threshold[1])]=1
    return binary_img

def mag_threshold(img,sobel_kernel=3,threshold=(0,255)):
    #gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray_img=img
    x_sobel=cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    y_sobel=cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    mag_sobel=np.sqrt(x_sobel**2+y_sobel**2)
    mag_sobel=np.uint8(255*mag_sobel/np.max(mag_sobel))
    binary_img=np.zeros_like(mag_sobel)
    binary_img[(mag_sobel>threshold[0])&(mag_sobel<threshold[1])]=1
    return binary_img

def dir_threshold(img,sobel_kernel=3,threshold=(0.0,np.pi/2)):
    #gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = img
    x_sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    x_sobel = np.abs(x_sobel)
    y_sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    y_sobel = np.abs(y_sobel)
    dir_sobel=np.arctan2(y_sobel,x_sobel)
    binary_img=np.zeros_like(dir_sobel,np.uint8)
    binary_img[(dir_sobel>threshold[0])&(dir_sobel<threshold[1])]=1
    return binary_img

def apply_threshold(img,threshold=(0,255)):
    binary_img = np.zeros_like(img,np.uint8)
    binary_img[(img>=threshold[0])&(img<=threshold[1])]=1
    return binary_img

def get_binary_image(undist_img):
    hls_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HLS)
    s_img = hls_img[:, :, 2]
    r_img = undist_img[:, :, 2]

    s_img = apply_threshold(s_img, (100, 255))  # 150,255
    r_img = apply_threshold(r_img, (150, 255))
    binary_img = s_img & r_img
    return binary_img

def warper(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    MInv = cv2.getPerspectiveTransform(dst, src)
    warped_binary = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return M, MInv, warped_binary

def get_lane_points_from_scratch(binary_img):
    left_lane_indices = []
    right_lane_indices = []
    nwindows = 9
    margin = 100
    minpix = 50
    histogram=np.sum(binary_img[binary_img.shape[0]//2:,:],axis=0)
    plt.plot(histogram)
    out_img = np.dstack((binary_img,binary_img,binary_img))*255
    mid_point = len(histogram)//2
    leftx_current = np.argmax(histogram[:mid_point])
    rightx_current = np.argmax(histogram[mid_point:])+mid_point
    yMax = binary_img.shape[0]
    xMax = binary_img.shape[1]
    print(xMax)
    print(yMax)
    window_height = yMax//nwindows
    nonzero = binary_img.nonzero()
    nonzeroX=nonzero[1]
    nonzeroY=nonzero[0]
    for i in range(nwindows):
        windowx_left_low = np.max([leftx_current-margin,0])
        windowx_left_high = leftx_current+margin
        windowx_right_low=rightx_current-margin
        windowx_right_high= np.min([rightx_current+margin, xMax])
        windowy_high = yMax - i*window_height
        windowy_low = yMax - (i+1)*window_height
        cv2.rectangle(out_img,(windowx_left_high,windowy_high),(windowx_left_low,windowy_low),(0,255,0),2)
        cv2.rectangle(out_img,(windowx_right_high,windowy_high),(windowx_right_low,windowy_low),(0,255,0),2)
        nonzeroIx=nonzeroX[(nonzeroX>windowx_left_low) & (nonzeroX<windowx_left_high) & (nonzeroY>windowy_low) & (nonzeroY<windowy_high)]
        nonzeroIy = nonzeroY[(nonzeroX > windowx_left_low) & (nonzeroX < windowx_left_high) & (nonzeroY > windowy_low) & (nonzeroY < windowy_high)]
        left_lane_indices.append((nonzeroIy,nonzeroIx))
        if(len(nonzeroIx)>minpix):
            leftx_current=np.int(np.average(nonzeroIx))
        nonzeroIx = nonzeroX[
            (nonzeroX > windowx_right_low) & (nonzeroX < windowx_right_high) & (nonzeroY > windowy_low) & (nonzeroY < windowy_high)]
        nonzeroIy = nonzeroY[
            (nonzeroX > windowx_right_low) & (nonzeroX < windowx_right_high) & (nonzeroY > windowy_low) & (nonzeroY < windowy_high)]
        right_lane_indices.append((nonzeroIy, nonzeroIx))
        if (len(nonzeroIx) > minpix):
            rightx_current = np.int(np.average(nonzeroIx))
    left_lane_indices=np.concatenate(left_lane_indices,axis=1)
    left_lane_indices=(np.array(left_lane_indices[0]),np.array(left_lane_indices[1]))
    right_lane_indices=np.concatenate(right_lane_indices,axis=1)
    right_lane_indices = (np.array(right_lane_indices[0]), np.array(right_lane_indices[1]))
    print(right_lane_indices)
    print(np.max(right_lane_indices[0]))
    print(np.max(right_lane_indices[1]))
    print(out_img.shape)
    left_fit=np.polyfit(left_lane_indices[0],left_lane_indices[1],2)
    right_fit=np.polyfit(right_lane_indices[0],right_lane_indices[1],2)
    curve_y=np.linspace(0,yMax-1,yMax)
    curve_x_left = np.array(left_fit[0]*curve_y**2+left_fit[1]*curve_y+left_fit[2], dtype= np.int32)
    left_curve = (np.array(curve_y,np.int32), curve_x_left)
    curve_x_right = np.array(right_fit[0]*curve_y**2+right_fit[1]*curve_y+right_fit[2], dtype= np.int32)
    right_curve = (np.array(curve_y,np.int32), curve_x_right)
    out_img[left_lane_indices]=[255,0,0]
    out_img[right_lane_indices]=[0,0,255]
    out_img[left_curve]=[255,255,0]
    out_img[right_curve]=[255,255,0]

    plt.figure()
    plt.imshow(out_img)




def process(img, mtx, dist):
    undist_img=cv2.undistort(img,mtx,dist,None,mtx)
    binary_img = get_binary_image(undist_img)
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[429, 560],
         [236, 690],
         [1073, 690],
         [865, 560]])

    dst = np.float32(
        [[(img_size[0] / 4), 0.85*img_size[1]],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0.85*img_size[1]]])
    M, MInv, warped_binary = warper(binary_img,src,dst)
    #cv2.rectangle(undist_img,tuple(src[0]),tuple(src[2]),(0,0,255), thickness=2)
    src = np.int32([src])
    cv2.polylines(undist_img,src,True,(0,0,255))
    warped_img=cv2.warpPerspective(undist_img,M,img_size,flags=cv2.INTER_LINEAR)
    get_lane_points_from_scratch(warped_binary)

    undist_img=cv2.cvtColor(undist_img,cv2.COLOR_BGR2RGB)
    warped_img=cv2.cvtColor(warped_img,cv2.COLOR_BGR2RGB)
    f,axes=plt.subplots(1,2)
    axes[0].imshow(undist_img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[1].imshow(warped_img,cmap='gray')
    axes[1].set_title("Warped Image")
    # axes[1,0].imshow(s_img,cmap='gray')
    # axes[1,0].set_title("S Image")
    # axes[1,1].imshow(r_img,cmap='gray')
    # axes[1,1].set_title("R Image")

    plt.show()