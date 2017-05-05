import os

import matplotlib.pyplot as plt


### feature extraction possibilites

# spatial bins  - resize image, flatten and use pixel value for features
# histogram colours - a concatenated vector of historgrams, grouped into x bins, for each colour channel
# HOG - computes a histogram of gradient directions within each cell block, binned in to x number of orientations (typically 9) - make sure you normalize

### Contents

# extract features from train images
# train a classifier using svm

# extract features from whole image
# perform sliding window search on image frame
# get features within window search frame
# pass features to trained svm and predict result
# if car detected record window points
# repeat until all windows have been search
# create heat map based on detected window points and threshold to remove spurious detections
# use outer edge of heatmap to draw bounds / figure out a better way of doing this
# use historical bounds to better predict future bounds

HOG = "hog"
SPATIAL = "spatial"
HISTOGRAM = "hist"

from utils import *

TEST_IMAGES_DIR = "./test_images"



if "__main__" == __name__:

    import pipeline

    test = False

    if test:

        for fname in [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR)]:

            if "" in fname:
                img = plt.imread(fname)
                draw_img = pipeline.process_image(img, fname=fname, save=True)

    else:

        from moviepy.editor import VideoFileClip

        white_output = 'project_video_out.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(pipeline.process_image)
        white_clip.write_videofile(white_output, audio=False)

