## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistortion_test5.jpg "Road Transformed"
[image3]: ./output_images/binary_image.png "Binary Example"
[image4]: ./output_images/warped_image.png "Warp Example"
[image5]: ./output_images/LaneLines.png "Fit Visual"
[image6]: ./output_images/Lane_Detected.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file "pipeline.py".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_p` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Now that I have the camera matrix and the distortion coefficients from the calibration step, I can use them to undistort images using the undistort function in cv2. The following is the result of applying unsitortion on one of the test images:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this part is in the file "pipeline.py" in the function "get_binary_image". I depended on two things:
1- A threshold on the magnetude of the gradient
2- Converting the image to the HLS color space and thresholding the S channel
The result of the two thresholding operations are then ORed. Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes two class functions: 'Lane.set_warp_matrix', and 'Lane.warper()'. The two functions appear in the file `pipeline.py`.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 429, 560      | 320, 612      | 
| 236, 690      | 320, 720      |
| 1073, 690     | 960, 720      |
| 865, 560      | 960, 612      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This was done in two functions:
1- 'Lane.__get_line_points_from_scratch()'. This is used to find the Lane lines from scratch without any previous knowledge of where the lanes should be.
2- 'Lane.__update_line_points()'. This is used when there is a previous knowledge of the lane lines.

Both functions take the binary image as an input. The first function uses a histogram to find a starting point to search for the lanes. Starting from these points, a square defining the area of search is used and updated to follow the lane lines from the bottom of the image to the top of it.

The seacond function uses the previously determined lane lines as a starting point and searches for the lanes in the vicinity of these lines.

Here is an example of the result.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is done in the class funtion "Lane.__get_radius_of_curvature". The radius of curvature is a function in the coefficients of the second order polinomial fit. The average of the last 10 frames is used in this function.
The position of the vehicle with respect to the center is calculated in the class function "Lane.update_vehicle_position()". This is done by calculating the X coordinates of the left and right lane lines at the bottom of the image and finding their average. The position of the vehicle is then the relative distance between the center of the lane and the center of the image (In the X direction). 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function "draw_lines()".  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the pipeline is not robust enough to brightness changes. It is not also robust enough against the change in the texture of the road. I think different color spaces could be used. A correction for brightness over the whole image as a preprocessing step could also be useful  
