

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

[image1]: ./Undistort_chart.jpg "Undistorted"
[image2]: ./Undistort_image.jpg "Road Transformed"
[image3]: ./combined_binary.jpg "Binary Example"
[image4]: ./color_binary.jpg "Warp Example"
[image5]: ./binary_warped.jpg "Warp Example"
[image6]: ./color_fit_lines.jpg "Fit Visual"
[image7]: ./example_output.jpg "Output"
[video1]: ./video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
(code line 39)
The code for this step is contained in lines 7 through 43 of the file called `p4.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
In the previous section, I calculated distortion matrix and made a `undistotImg()` function to undistort an image.  I applied this function to the test image and obtained this result:

![alt text][image2]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 64 through 106 in `p4.py`).  Here's an example of my output for this step.

![alt text][image3]

This binary image consists of the following three binary images.
1. S channels of HLS and HSV color spaces. In HLS space, lane lines are represented by high s values. In HSV space, s channel is similar to one in HLS space. However, white pixels are represented by low s values. Therefore I subtract s in HSV from s in HSL to extract white lane lines.
2. Yellow channel. I made it by red and green channels. In addition, I subtract blue channel from yellow channel to enhance yellow line.
3. Gradient of red and blue channels.  

Here is an example of each components. (1:red, 2:green, 3:blue)

![alt text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 146 through 161 in the file `p4.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
offset_dst=300
src = np.float32(
    [[offset_dst, 0],
    [img_size[0]-offset_dst, 0],
    [img_size[0]-offset_dst, img_size[1]],
    [offset_dst, img_size[1]]])

src_center=[640, 655]
width=[80, 460]
dst = np.float32(
    [[src_center[0]-width[0], 470],
    [src_center[0]+width[0], 470],
    [src_center[1]+width[1], img_size[1]],
    [src_center[1]-width[1], img_size[1]]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 560, 470      | 300, 0        |
| 720, 470      | 980, 0        |
| 1115, 720     | 980, 720      |
| 185, 720      | 300, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in lines 186 through 279 of the file called `p4.py`.
For the first frame, I fit my lane lines with a 2nd order polynomial using the `detectLines()` function like this:

![alt text][image6]

After second frame, I searched just a window around the previous detection to improve speed and provide a more robust method for rejecting outliers.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 324 through 345 in my code in `p4.py`.
I estimated how much the road is curving in `detCurvature()`.The radius of curvature is given in meters assuming the curve of the road follows a circle.  
I estimated where the vehicle is located with respect to the center of the lane in `getCarPos()`. For the position of the vehicle, I assume the camera is mounted at the center of the car.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 366 through 393 in my code in `p4.py` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my video result.   
 [video_out.mp4](./video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
1. Noise.  
I used gradient, so if noise is contained in image, detected line will also be noisy and estimated curvature will contain the error. To avoid this, I could smooth images like low-pass filtering to mitigate the noise.
2. Slope.  
I assumed the road is flat and chose the hardcode the source and destination points in step 3. If the road slopes, these points should be changed according to the inclination of the road.
3. Occlusion and Low contrast.  
If the road is occluded by other vehicle or the contrast between lines and road surface is very low due to sunlight and shadow, I cannot recognize lane lines. I used the fitting information of the previous frame so if I fail to find lines in some frames successively my pipeline cannot recover. I should calculate how confident the line detection are and if the confidence is low I ignore the fitting of the previous frame to recover from the failure.  
4. Fake edges.  
If there are some edges along the lane lines because of shadow or road structures, my pipeline will interpret it as lane lines. To avoid this, I should judge which the optimal lines are by the information about the lane width or the possible range of lane curvatures and car positions.
