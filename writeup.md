##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step in the extract\_hog\_features() function, which is a wrapper for get\_hog\_features() on lines 102 and 182, respectively, in utils.py

I started by reading in the `vehicle` images from the KITTI suite. I discarded the GTI images as these contained near duplicate images and caused the model to over fit. I limited the number `non-vehicle` images to 6000 to equal that of the `vehicle` images, to avoid non-vehicle bias.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car Image
![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Vehicle-Detection/master/examples/example_car.jpg)

Not Car Image
![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Vehicle-Detection/master/examples/example_noncar.jpg)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Y` channel in the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Vehicle-Detection/master/examples/hog_image_Ychannel.jpg)

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2) performed well on the test set (~ 99%) and was able to train in < 20 seconds and build features in < 60 seconds.

There was no significant improvement choosing parameters which provided more features, e.g more orientations or fewer pixels\_per\_cell and the increased learning and extraction time made these options prohibitive.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The linear SVM was trained in `train.py` file. The features for the SVM were extracted in lines 59 - 73, based on the params argument passed to the function.

The parameters are defined on line 20 in the `params dict` on line 20 in `pipeline.py`. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implement in lines ~ 105 to 180 in `pipeline.py`. 

The sliding window search starts from halfway down the image, as on a flat road vehicles would not be above this point.

The sliding window searches over 3 scales: 1.5, 2. as this typically covers all sizes of vehicles. 

I chose a cell step of 1 to maxmise the chances of detecting a vehicle. This resulted in more detections being made, a 'hotter' heatmap, allowing a higher threshold to be applied, which results in more false positives being removed.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features with a 1 cell step, which provided a nice result.  Here are some example images:

Boxes found after sliding window search:
![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Vehicle-Detection/master/output_images/boxed_test1.jpg)

Heatmap applied to above image:
![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Vehicle-Detection/master/output_images/heat_test1.jpg)

Final output:
![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Vehicle-Detection/master/output_images/final_test1.jpg)

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://raw.githubusercontent.com/dpdenton/CarND-Vehicle-Detection/master/vide/project_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The SVM was training well and produce a result of > 99% accuracy on the test set. This transalted reasonably well to the test images and the videos. A few false positives were detected, however this is to be expected give its making approx 5000 predictions every image. The more concerning issue is, at times, it doesn't generate enough detections when the window is over a vehicle. This apeared to be a problem when a vehicle is partially in shot, or when it's at a wide angle and the rear of the vehicle is at a significant angle in relation to the camera. This may be due to the `vehicle` images in the training data be prodominantly of the rear of vehicles, and few at any significant angle.

The bounding boxes in my model weren't particularly accurate. Whilst they detected the vehicle correctly, they often only partially bound the vehicle, or there was too much space between the vehicle and the bound. 

To improve the pipeline I'd investigate developing a basic, relatively big non-overlapping 'high-step' sliding window to quickly scan the frame and detect potential vehicles, then pass those detections to a more sophisticated model, like a conv neural net, to determine if it's actually a vehicle, and if it is, provide a more accurate bound for the vehicle.

I would also look to improve the tranining data to provide more car angles, as the detections tend to be more focusses on the rear of a vehicle, perhaps augmenting the data with perspective transforms or additional images of angled vehicles.
