# Lane-Detection-and-folowing-with-Intel-Integrated-Primitives-IPP-library
This code is a lane detection and following algorithm that is completes image processing techniques in parallel computing with a help of IPP.
There are various types of Lane Detection and each of them has its advantages and disadvantages.
We used several algorithms to detect the lane:
1) Image denoising.
2) Edge detection from binary image
3) Mask the image
4) Hough lines detection
5) Left and right lines separation
6) Line regressing 
7) Drawing the complete line


First, we denoise the image using Gaussian blur function, since it’s the fastest and most useful filter. 
Gaussian filtering is done by convolving each point in the input array with a Gaussian kernel and then summing them all to produce the output array.

Then we used edge detection to find boundaries of objects in the image
Before detecting the edges, we convert the 3-channel color image to 1 channel gray image.
We apply binarization technique to reduce the features from image
Then using 2D filter we find the edges.

Masking the image, ROI

The idea is that we recalculate each pixel's value in an image according to a mask matrix. From a mathematical point of view we make a weighted average, with our specified values.
We mask the image in order to take the proper Region of Interest.
Image size is 1920x1080px
We mask the half size of the image and take only the lower part and we use the edges detected there.

Hough lines	and line separation
We got the Hough lines from the detected edges (4 points) and store all those lines in vector variable.
As soon as we get all the points, we separate them into those which lay on left side and right side. 


As soon as we separate all the lines, we do the regression of the lines and merge the several close lines into one bigger lines and illuminate other lines
At the end we get two thick lines 
Then we plot them on an original image 

Result



![Alt text](/input.png)


