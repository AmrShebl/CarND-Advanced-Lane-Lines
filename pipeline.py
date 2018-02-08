import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

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
    gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #gray_img=img
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
    #r_img = undist_img[:, :, 2]
    mag_img = mag_threshold(undist_img,5,(50,255))

    s_img = apply_threshold(s_img, (150, 255))  # 150,255
    #r_img = apply_threshold(r_img, (50, 255))
    binary_img = s_img | mag_img
    return binary_img


class Line():
    def __init__(self, n_average=10):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        #self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        #self.bestx = None
        # polynomial coefficients of the last n iterations
        self.last_n_fit_coefficients = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [0, 0, 0]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.all_indices=(np.array(0,np.int32),np.array(0,np.int32))
        self.n_average=n_average
        self.n_sucessive_failure=0
    def update_line_coefficient(self, line_coefficients):
        self.diffs=line_coefficients-self.current_fit
        self.current_fit=line_coefficients
        if(len(self.last_n_fit_coefficients)<self.n_average):
            self.last_n_fit_coefficients.append(line_coefficients)
        else:
            self.last_n_fit_coefficients=self.last_n_fit_coefficients[1:]
            self.last_n_fit_coefficients.append(line_coefficients)
    def get_best_fit(self):
        if len(self.last_n_fit_coefficients) is 0:
            return None
        return np.average(self.last_n_fit_coefficients, axis=0)
    def should_we_reset(self):
        return self.n_sucessive_failure>=self.n_average or len(self.last_n_fit_coefficients)<self.n_average

class Lane():
    def __init__(self):
        self.left_line=Line()
        self.right_line=Line()
        self.__camera_matrix=None
        self.__camera_distortion=None
        self.__warp_matrix=None
        self.__inverse_warp_matrix=None
        self.__vehicle_position=0

    def __get_radius_of_curvature(self,curve_x,curve_y, y_eval):
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        fit=np.polyfit(curve_y*ym_per_pix,curve_x*xm_per_pix,2)
        return ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    def update_vehicle_position(self, yMax, xMax):
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        left_fit = self.left_line.get_best_fit()
        right_fit = self.right_line.get_best_fit()
        x_left = left_fit[0]*yMax**2+left_fit[1]*yMax+left_fit[2]
        x_right = right_fit[0]*yMax**2+right_fit[1]*yMax+right_fit[2]
        x_center = 0.5*(x_left+x_right)
        self.__vehicle_position = (0.5*xMax - x_center)*xm_per_pix

    def calibrate_camera(self, dir, n_corners):
        images = glob.glob(dir)
        obj_points = []
        img_points = []
        obj_p = np.zeros((n_corners[0] * n_corners[1], 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:n_corners[0], 0:n_corners[1]].T.reshape(-1, 2)
        img_size = 0
        for imagef in images:
            img = cv2.imread(imagef)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = gray_img.shape[::-1]
            ret, img_p = cv2.findChessboardCorners(gray_img, n_corners)
            if ret:
                obj_points.append(obj_p)
                img_points.append(img_p)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
        self.__camera_matrix=mtx
        self.__camera_distortion=dist

    def warper(self,img):
        img_size = (img.shape[1], img.shape[0])
        warped_binary = cv2.warpPerspective(img, self.__warp_matrix, img_size, flags=cv2.INTER_LINEAR)
        return warped_binary

    def set_warp_matrix(self,src,dst):
        self.__warp_matrix = cv2.getPerspectiveTransform(src, dst)
        self.__inverse_warp_matrix = cv2.getPerspectiveTransform(dst, src)

    def __get_line_failed(self):
        self.left_line.detected = False
        self.right_line.detected = False
        self.left_line.n_sucessive_failure+=1
        self.right_line.n_sucessive_failure+=1


    def __get_line_points_from_scratch(self,binary_img):
        #plt.imshow(binary_img, cmap='gray')
        #plt.show()
        ret = True
        left_lane_indices = []
        right_lane_indices = []
        nwindows = 9
        margin = 100
        minpix = 50
        histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)
        plt.plot(histogram)
        out_img = np.dstack((binary_img, binary_img, binary_img)) * 255
        mid_point = len(histogram) // 2
        leftx_current = np.argmax(histogram[:mid_point])
        rightx_current = np.argmax(histogram[mid_point:]) + mid_point
        yMax = binary_img.shape[0]
        xMax = binary_img.shape[1]
        window_height = yMax // nwindows
        nonzero = binary_img.nonzero()
        nonzeroX = nonzero[1]
        nonzeroY = nonzero[0]
        for i in range(nwindows):
            windowx_left_low = np.max([leftx_current - margin, 0])
            windowx_left_high = leftx_current + margin
            windowx_right_low = rightx_current - margin
            windowx_right_high = np.min([rightx_current + margin, xMax])
            windowy_high = yMax - i * window_height
            windowy_low = yMax - (i + 1) * window_height
            cv2.rectangle(out_img, (windowx_left_high, windowy_high), (windowx_left_low, windowy_low), (0, 255, 0), 2)
            cv2.rectangle(out_img, (windowx_right_high, windowy_high), (windowx_right_low, windowy_low), (0, 255, 0), 2)
            nonzeroIx = nonzeroX[
                (nonzeroX > windowx_left_low) & (nonzeroX < windowx_left_high) & (nonzeroY > windowy_low) & (
                nonzeroY < windowy_high)]
            nonzeroIy = nonzeroY[
                (nonzeroX > windowx_left_low) & (nonzeroX < windowx_left_high) & (nonzeroY > windowy_low) & (
                nonzeroY < windowy_high)]
            left_lane_indices.append((nonzeroIy, nonzeroIx))
            if (len(nonzeroIx) > minpix):
                leftx_current = np.int(np.average(nonzeroIx))
            nonzeroIx = nonzeroX[
                (nonzeroX > windowx_right_low) & (nonzeroX < windowx_right_high) & (nonzeroY > windowy_low) & (
                nonzeroY < windowy_high)]
            nonzeroIy = nonzeroY[
                (nonzeroX > windowx_right_low) & (nonzeroX < windowx_right_high) & (nonzeroY > windowy_low) & (
                nonzeroY < windowy_high)]
            right_lane_indices.append((nonzeroIy, nonzeroIx))
            if (len(nonzeroIx) > minpix):
                rightx_current = np.int(np.average(nonzeroIx))
        left_lane_indices = np.concatenate(left_lane_indices, axis=1)
        left_lane_indices = (np.array(left_lane_indices[0]), np.array(left_lane_indices[1]))
        right_lane_indices = np.concatenate(right_lane_indices, axis=1)
        right_lane_indices = (np.array(right_lane_indices[0]), np.array(right_lane_indices[1]))
        if len(left_lane_indices[0])==0 or len(right_lane_indices[0])==0:
            self.__get_line_failed()
            return False
        left_fit = np.polyfit(left_lane_indices[0], left_lane_indices[1], 2)
        right_fit = np.polyfit(right_lane_indices[0], right_lane_indices[1], 2)
        curve_y = np.linspace(0, yMax - 1, yMax)
        curve_x_left = np.array(left_fit[0] * curve_y ** 2 + left_fit[1] * curve_y + left_fit[2], dtype=np.int32)
        left_curve = (np.array(curve_y, np.int32), curve_x_left)
        max_value = np.max(np.abs(curve_x_left))
        if max_value>=xMax:
            self.__get_line_failed()
            return False
        curve_x_right = np.array(right_fit[0] * curve_y ** 2 + right_fit[1] * curve_y + right_fit[2], dtype=np.int32)
        max_value = np.max(np.abs(curve_x_right))
        if max_value >= xMax:
            self.__get_line_failed()
            return False
        right_curve = (np.array(curve_y, np.int32), curve_x_right)
        if ret:
            out_img[left_lane_indices] = [255, 0, 0]
            out_img[right_lane_indices] = [0, 0, 255]
            out_img[left_curve] = [255, 255, 0]
            out_img[right_curve] = [255, 255, 0]



        left_radius_of_curvature = self.__get_radius_of_curvature(left_lane_indices[1], left_lane_indices[0],yMax)
        right_radius_of_curvature = self.__get_radius_of_curvature(right_lane_indices[1], right_lane_indices[0],yMax)
        print("The left radius of curvature is {} and the right is {}".format(left_radius_of_curvature,
                                                                              right_radius_of_curvature))

        if(left_fit[0]*right_fit[0]<0):
            self.__get_line_failed()
            return False

        # f, (ax0, ax1) = plt.subplots(1,2)
        # ax0.imshow(binary_img, cmap='gray')
        # ax0.set_title('Binary Image')
        # ax1.imshow(out_img)
        # ax1.set_title('Lane Lines')
        # plt.show()

        if(ret):
            self.left_line.detected = True
            self.right_line.detected = True
            self.left_line.all_indices = left_lane_indices
            self.right_line.all_indices = right_lane_indices
            self.left_line.update_line_coefficient(left_fit)
            self.right_line.update_line_coefficient(right_fit)
            self.left_line.radius_of_curvature=left_radius_of_curvature
            self.right_line.radius_of_curvature=right_radius_of_curvature
            self.left_line.n_sucessive_failure=0
            self.right_line.n_sucessive_failure=0
            self.update_vehicle_position(yMax,xMax)
            print(self.__vehicle_position)

        return ret


    def __update_line_points(self,binary_img):
        ret = True
        nonzero=binary_img.nonzero()
        nonzeroX=nonzero[1]
        nonzeroY=nonzero[0]
        margin=100
        yMax=binary_img.shape[0]
        xMax=binary_img.shape[1]
        left_fit = self.left_line.get_best_fit()
        right_fit = self.right_line.get_best_fit()
        #left_fit = self.left_line.current_fit
        #right_fit = self.right_line.current_fit
        left_lane_inds = ((nonzeroX > (left_fit[0] * (nonzeroY ** 2) + left_fit[1] * nonzeroY +
                                       left_fit[2] - margin)) & (nonzeroX < (left_fit[0] * (nonzeroY ** 2) +
                                                                             left_fit[1] * nonzeroY + left_fit[
                                                                                 2] + margin)))

        right_lane_inds = ((nonzeroX > (right_fit[0] * (nonzeroY ** 2) + right_fit[1] * nonzeroY +
                                        right_fit[2] - margin)) & (nonzeroX < (right_fit[0] * (nonzeroY ** 2) +
                                                                               right_fit[1] * nonzeroY + right_fit[
                                                                                   2] + margin)))
        leftx = nonzeroX[left_lane_inds]
        lefty = nonzeroY[left_lane_inds]
        rightx = nonzeroX[right_lane_inds]
        righty = nonzeroY[right_lane_inds]
        left_lane_indices = (np.array(lefty,np.int32),np.array(leftx,np.int32))
        right_lane_indices = (np.array(righty,np.int32),np.array(rightx,np.int32))

        if len(left_lane_indices[0])==0 or len(right_lane_indices[0])==0:
            self.__get_line_failed()
            return False
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        out_img = np.dstack((binary_img, binary_img, binary_img)) * 255
        curve_y = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        curve_x_left = left_fit[0] * curve_y ** 2 + left_fit[1] * curve_y + left_fit[2]
        max_value = np.max(np.abs(curve_x_left))
        if max_value >= xMax:
            self.__get_line_failed()
            return False
        curve_x_right = right_fit[0] * curve_y ** 2 + right_fit[1] * curve_y + right_fit[2]
        max_value = np.max(np.abs(curve_x_right))
        if max_value >= xMax:
            self.__get_line_failed()
            return False

        left_curve = (np.array(curve_y, np.int32), np.array(curve_x_left,np.int32))
        right_curve = (np.array(curve_y, np.int32), np.array(curve_x_right,np.int32))

        out_img[left_lane_indices] = [255, 0, 0]
        out_img[right_lane_indices] = [0, 0, 255]
        out_img[left_curve] = [255, 255, 0]
        out_img[right_curve] = [255, 255, 0]

        plt.figure()
        plt.imshow(out_img)

        left_radius_of_curvature = self.__get_radius_of_curvature(leftx, lefty, yMax)
        right_radius_of_curvature = self.__get_radius_of_curvature(rightx, righty, yMax)
        print("The left radius of curvature is {} and the right is {}".format(left_radius_of_curvature,right_radius_of_curvature))

        if (left_fit[0] * right_fit[0] < 0):
            self.__get_line_failed()
            return False


        if ret:
            self.left_line.detected = True
            self.right_line.detected = True
            self.left_line.all_indices = left_lane_indices
            self.right_line.all_indices = right_lane_indices
            self.left_line.update_line_coefficient(left_fit)
            self.right_line.update_line_coefficient(right_fit)
            self.left_line.radius_of_curvature=left_radius_of_curvature
            self.right_line.radius_of_curvature=right_radius_of_curvature
            self.update_vehicle_position(yMax, xMax)
            print(self.__vehicle_position)
        return ret

    def __draw_lane(self,warped_img, shape):
        curve_y = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        left_fitx=self.left_line.get_best_fit()
        right_fitx=self.right_line.get_best_fit()
        if (left_fitx is None) or (right_fitx is None):
            return None
        left_curve_x=left_fitx[0]*curve_y**2+left_fitx[1]*curve_y+left_fitx[2]
        right_curve_x=right_fitx[0]*curve_y**2+right_fitx[1]*curve_y+right_fitx[2]
        pts_left = np.array([np.transpose(np.vstack([left_curve_x, curve_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_curve_x, curve_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.__inverse_warp_matrix, (shape[1], shape[0]))
        return newwarp



    def process_image(self,img, force_reset=False):
        undist_img = cv2.undistort(img, self.__camera_matrix, self.__camera_distortion, None, self.__camera_matrix)
        output_img = np.copy(undist_img)
        binary_img = get_binary_image(undist_img)
        img_size = (img.shape[1], img.shape[0])
        src = np.float32(
            [[429, 560],
             [236, 690],
             [1073, 690],
             [865, 560]])

        dst = np.float32(
            [[(img_size[0] / 4), 0.85 * img_size[1]],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0.85 * img_size[1]]])

        self.set_warp_matrix(src,dst)
        warped_binary = self.warper(binary_img)
        # cv2.rectangle(undist_img,tuple(src[0]),tuple(src[2]),(0,0,255), thickness=2)
        src = np.int32([src])
        cv2.polylines(undist_img, src, True, (0, 0, 255))
        warped_img = cv2.warpPerspective(undist_img, self.__warp_matrix, img_size, flags=cv2.INTER_LINEAR)
        print(self.left_line.should_we_reset())
        if self.left_line.should_we_reset() or self.right_line.should_we_reset() or force_reset :
            self.__get_line_points_from_scratch(warped_binary)
        else:
            self.__update_line_points(warped_binary)


        newwarp=self.__draw_lane(warped_binary,undist_img.shape)

        # Combine the result with the original image
        if newwarp is not None:
            result = cv2.addWeighted(output_img, 1, newwarp, 0.3, 0)
        else:
            result = output_img
        #result = np.dstack((255*binary_img, 255*binary_img, 255*binary_img))

        #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


        # plt.imshow(result)
        #
        #
        # undist_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB)
        # warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
        # f, axes = plt.subplots(1, 2)
        # axes[0].imshow(undist_img, cmap='gray')
        # axes[0].set_title("Original Image")
        # axes[1].imshow(warped_img, cmap='gray')
        # axes[1].set_title("Warped Image")
        #
        # plt.show()
        return result


