import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#### 1. calibrate camera ####
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

#calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#undistort function
def undistotImg(img_in):
    img_out= cv2.undistort(img_in, mtx, dist, None, mtx)
    return img_out
# Plotting undistorted image
img_test = mpimg.imread(images[0])
img_undistorted=undistotImg(img_test)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
ax1.set_title(' Original image')
ax1.imshow(img_test)
ax2.set_title('Undistorted image')
ax2.imshow(img_undistorted)
f.savefig('output_images/Undistort_chart.jpg')

images_test = glob.glob('test_images/*.jpg')
img_test = mpimg.imread(images_test[3])
img_undistorted=undistotImg(img_test)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
ax1.set_title(' Original image')
ax1.imshow(img_test)
ax2.set_title('Undistorted image')
ax2.imshow(img_undistorted)
f.savefig('output_images/Undistort_image.jpg')

#### 2. thresholding ####
#Applying Sobel filter and Thresholding (sub-function of the 'thresholding' function)
def sobel_thresh(img, thresh=[40,255],ksize=5):
    # Sobel x
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sxbinary = np.zeros((sobelx.shape[0], sobelx.shape[1]))
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary
# Thresholding
def thresholding(img,thresh=[50,150], s_thresh_min=65, y_thresh_min=35,ksize=5):
    # Convert to HLS and HSV color spaces to extract white lines
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2HSV)
    s_channel = hls[:,:,2]*1.2-hsv[:,:,1] #subtract and balance the s values
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min)] = 1

    #Convert to yellow channel and subtract blue to extract yellow lines
    yellow_channel=np.sqrt((img[:,:,0]/255)**2+(img[:,:,1]/255)**2)*255.0
    yellow_enhanced=yellow_channel-img[:,:,2]*1.7 #yellow-blue
    y_binary = np.zeros_like(yellow_enhanced)
    y_binary[(yellow_enhanced >= y_thresh_min)] = 1

    # Threshold red and blue channel's gradients
    sxbinary_col=np.zeros_like(img)
    for c_channel in [0,2]:
        sxbinary_col[:,:,c_channel]=sobel_thresh(img[:,:,c_channel],thresh, ksize)
    sxbinary=((sxbinary_col[:,:,0] == 1) | (sxbinary_col[:,:,1] == 1) | (sxbinary_col[:,:,2] == 1)).astype('float32')

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( y_binary, s_binary, sxbinary))

    # Combine the three binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(y_binary == 1) | (s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary, color_binary
# Plotting thresholded images
[combined_binary, color_binary]=thresholding(img_undistorted)
plt.imshow(combined_binary, cmap='gray')
plt.show()
plt.savefig('output_images/combined_binary.jpg')
plt.imshow(color_binary)
plt.show()
plt.savefig('output_images/color_binary.jpg')
#for test (yellow line)
for i in range(10):
    img_in = mpimg.imread(images_test[i])
    img_undistorted=undistotImg(img_in)
    [combined_binary, color_binary]=thresholding(img_undistorted,[50,150],65,35,ksize=5)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,8))
    ax1.set_title('Stacked thresholds:'+str(i))
    ax1.imshow(color_binary)
    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')

#for test (yellow line)
yellow_channel=np.sqrt((img_undistorted[:,:,0]/255)**2+(img_undistorted[:,:,1]/255)**2)*255.0
img=yellow_channel-img_undistorted[:,:,2]*1.7
f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,12))
ax1.imshow(yellow_channel)
ax2.imshow(img)
ax3.imshow(sobel_thresh(img, thresh=[15,240],ksize=5))

#for test (white line)
for image_test in images_test:
    img_in = mpimg.imread(image_test)
    img_undistorted=undistotImg(img_in)
    hls = cv2.cvtColor(img_undistorted, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2HSV)
    img=hls[:,:,2]*1.2-hsv[:,:,1]
    f,(ax1,ax2, ax3)=plt.subplots(1,3, figsize=(12,10))
    ax1.imshow(img)
    ax3.imshow(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
    ax2.imshow(sobel_thresh(img, thresh=[25,255],ksize=5))

#### 3. perpective transform ####
def warper(img):
    offset_dst=300
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    src=determine_src(img)
    dst = np.float32([[offset_dst, 0], [img_size[0]-offset_dst, 0], [img_size[0]-offset_dst, img_size[1]], [offset_dst, img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv, dst
def determine_src(img_in):
    img_size = (img.shape[1], img.shape[0])
    src_center=[640, 655]
    width=[80, 460]
    src = np.float32([[src_center[0]-width[0], 470], [src_center[0]+width[0], 470], [src_center[1]+width[1], img_size[1]], [src_center[1]-width[1], img_size[1]]])
    return src

# Plotting transformed images
def plotTransformedImage(combined_binary):
    binary_warped,Minv, dst=warper(combined_binary)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
    ax1.set_title('input image')
    ax1.imshow(combined_binary, cmap='gray')
    src=determine_src(combined_binary)
    ax1.plot(src[[0,1,2,3,0],0],src[[0,1,2,3,0],1])
    ax2.set_title('transformed image')
    ax2.imshow(binary_warped, cmap='gray')
    ax2.plot(dst[[0,1,2,3,0],0],dst[[0,1,2,3,0],1])
    return binary_warped


for i in [3]:
    img_in = mpimg.imread(images_test[i])
    img_undistorted=undistotImg(img_in)
    [combined_binary, color_binary]=thresholding(img_undistorted)
    binary_warped=plotTransformedImage(combined_binary)
    plt.savefig('output_images/binary_warped.jpg')


#### 4. detect lanes ####
def detectLines(binary_warped, nonzero, minpix = 50, nwindows = 18, margin = 100):
        # margin: Set the width of the windows +/- margin
        # minpix: Set minimum number of pixels found to recenter window
        # nwindows: Choose the number of sliding windows
    # Identify the x and y positions of all nonzero pixels in the image
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return left_lane_inds, right_lane_inds

def fitLine(binary_warped):
    global left_fit, right_fit
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left_fit == []:
        margin = 100
        left_lane_inds, right_lane_inds=detectLines(binary_warped, nonzero, margin=margin)
    else:
        margin = 30
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if (len(leftx)*len(lefty)*len(rightx)*len(righty))!=0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
        right_fit_new = np.polyfit(righty, rightx, 2)

        if left_fit == []:
            left_fit=left_fit_new
            right_fit=right_fit_new
        else:
            # smoothing using previous values
            left_fit=left_fit_new*(1.0/3.0)+left_fit*(2.0/3.0)
            right_fit=right_fit_new*(1.0/3.0)+right_fit*(2.0/3.0)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, margin

## plotting histgram
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
plt.figure()
plt.plot(histogram)
## plotting detect area
global left_fit, right_fit
left_fit=[]
right_fit=[]
left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, margin = fitLine(binary_warped)
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Create an image to draw on and an image to show the selection window
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.figure()
plt.imshow(result)
plt.plot(leftx, lefty, 'o', color='red')
plt.plot(rightx, righty, 'o', color='red')
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.savefig('output_images/color_fit_lines.jpg')

#### 5. determine the lane curvature ####
def detCurvature(ploty, left_fitx, right_fitx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad
def getCarPos(left_fitx,right_fitx,Xsize=1280):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    center_x=(left_fitx[0]+right_fitx[0])/2
    return (center_x-Xsize/2)*xm_per_pix

#### 6. visualize results ####
# Create an image to draw the lines on
def outputResult(binary_warped,Minv,left_fitx,right_fitx,ploty):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))
    return newwarp


#### test whole pipeline ####
def process_image(img_in):
    global left_fit, right_fit
    img_undistorted=undistotImg(img_in)
    [combined_binary, color_binary]=thresholding(img_undistorted)
    binary_warped,Minv, dst=warper(combined_binary)
    left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, margin = fitLine(binary_warped)
    left_curverad, right_curverad=detCurvature(ploty, left_fitx, right_fitx)
    newwarp=outputResult(binary_warped,Minv,left_fitx,right_fitx,ploty)
    result = cv2.addWeighted(img_undistorted, 1, newwarp, 0.3, 0)
    cv2.putText(result,'curvature:%d m(left), %d m(right)' % (left_curverad ,right_curverad),(10,50),cv2.FONT_HERSHEY_DUPLEX, 1.5,(100,255,0),2, cv2.LINE_AA)
    centerPos=getCarPos(left_fitx,right_fitx)
    if centerPos>0:
        cv2.putText(result,'Vehicle is %.2f m left of center' % centerPos,(10,100),cv2.FONT_HERSHEY_DUPLEX, 1.5,(100,255,0),2, cv2.LINE_AA)
    else:
        cv2.putText(result,'Vehicle is %.2f m right of center' % -centerPos,(10,100),cv2.FONT_HERSHEY_DUPLEX, 1.5,(100,255,0),2, cv2.LINE_AA)
    return result

for i in [3]:
    img_in = mpimg.imread(images_test[i])
    global left_fit, right_fit
    left_fit=[]
    right_fit=[]
    img_out=process_image(img_in)
    plt.title(image_test)
    plt.imshow(img_out)
    plt.show()
    plt.savefig('output_images/example_output.jpg')

#### output video ####

from moviepy.editor import VideoFileClip

video_output = 'video_out7.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip_out = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip_out.write_videofile(video_output, audio=False, threads=2)

video_output_challenge = 'video_out_challenge2.mp4'
clip2 = VideoFileClip("challenge_video.mp4")
clip_out_challenge = clip2.fl_image(process_image) #NOTE: this function expects color images!!
clip_out_challenge.write_videofile(video_output_challenge, audio=False, threads=2)


#check a specific frame
def movie_frame(time_s=10):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,10))
    ax1.imshow(clip1.make_frame(time_s))
    ax2.imshow(clip_out.make_frame(time_s))
movie_frame(40.9)

cv2.imwrite('test7.jpg',cv2.cvtColor(clip1.make_frame(22.53),cv2.COLOR_RGB2BGR))
cv2.imwrite('test8.jpg',cv2.cvtColor(clip1.make_frame(41.9),cv2.COLOR_RGB2BGR))
