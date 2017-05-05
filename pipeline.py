

import numpy as np

import train
from utils import *


HOG = "hog"
SPATIAL = "spatial"
HISTOGRAM = "hist"


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    print("Found {} bounding boxees".format(len(bbox_list)))
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # sanity check

        width = bbox[1][0] - bbox[0][0]
        height = bbox[1][1] - bbox[0][1]

        # height greater than width
        if height > 1.5 * width:
            continue

        # not on road surface
        if bbox[1][1] < 460:
            continue

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

heatmaps = []


def process_image(img, fname="video.jpg", save=False):

    # define hyperparams here
    # if a key doesn't exist those features will not be extracted and used to train the model
    params = {
        "ystart": int(img.shape[0]/2),
        "ystop": int(img.shape[0]),
        "scales": [1.5, 2],
        "features": {
            HOG: {
                "hog_channel": "ALL", # options 0, 1, 2, "ALL"
                "cspace": "YCrCb",
                "orient": 9,
                "pix_per_cell": 8,
                "cell_per_block": 2,
            },
            # SPATIAL: {
            #     "fn": extract_bin_spatial,
            #     "size": (32, 32),
            # },
            # HISTOGRAM: {
            #     "fn": extract_color_hist,
            #     "nbins": 32,
            #     "bins_range": (0, 256),
            # }
        }
    }

    print("Using {}".format(params))

    svc, X_scaler = train.run(params, max_sample_size=None, retrain=False, save=True)

    bboxes = []

    # get params
    ystart = params["ystart"]
    ystop = params["ystop"]
    scale = params["scales"]
    orient = params["features"][HOG]["orient"]
    pix_per_cell = params["features"][HOG]["pix_per_cell"]
    cell_per_block = params["features"][HOG]["cell_per_block"]
    cspace = params["features"][HOG]["cspace"]

    draw_img = np.copy(img)
    original_img = np.copy(img)

    # convert colour space and normalize (glob is already between 0-1)
    img = convert_color(img, cspace)
    img = img.astype(np.float64) / 255

    ctrans_tosearch = img[ystart:ystop, :, :]

    imshape = ctrans_tosearch.shape
    # scale 1
    scale1 = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / 1.5), np.int(imshape[0] / 1.5)))
    scale2 = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / 2), np.int(imshape[0] / 2)))

    scale1_ch1 = scale1[:, :, 0]
    scale1_ch2 = scale1[:, :, 1]
    scale1_ch3 = scale1[:, :, 2]

    scale2_ch1 = scale2[:, :, 0]
    scale2_ch2 = scale2[:, :, 1]
    scale2_ch3 = scale2[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (scale1_ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (scale1_ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    scale1_hog1 = get_hog_features(scale1_ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    scale1_hog2 = get_hog_features(scale1_ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    scale1_hog3 = get_hog_features(scale1_ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Compute individual channel HOG features for the entire image
    scale2_hog1 = get_hog_features(scale2_ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    scale2_hog2 = get_hog_features(scale2_ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    scale2_hog3 = get_hog_features(scale2_ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # init feature arrays
            features = np.empty((0))

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            if HOG in params['features']:

                # Extract HOG for this patch
                scale1_hog_feat1 = scale1_hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                scale1_hog_feat2 = scale1_hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                scale1_hog_feat3 = scale1_hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                features = np.hstack((features, scale1_hog_feat1, scale1_hog_feat2, scale1_hog_feat3))

                if xpos + nblocks_per_window < scale2_ch1.shape[1]:

                    scale2_hog_feat1 = scale2_hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    scale2_hog_feat2 = scale2_hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    scale2_hog_feat3 = scale2_hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                    scale2_features = np.hstack((scale2_hog_feat1, scale2_hog_feat2, scale2_hog_feat3))

                    if features.shape == scale2_features.shape:
                        features = np.vstack((features, scale2_features))

            if SPATIAL in params['features']:

                # Get color features
                spatial_features = bin_spatial(subimg, **params['features'][SPATIAL])
                features = np.hstack((features, spatial_features))

            if HISTOGRAM in params['features']:

                hist_features = color_hist(subimg, **params['features'][HISTOGRAM])
                features = np.hstack((features, hist_features))

            # Scale features and make a prediction
            # if a 1D vector reshape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            test_features = X_scaler.transform(features)
            #test_features = X_scaler.transform(features)
            # original
            # test_features = X_scaler.transform(np.hstack((features)).reshape(1, -1))

            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_predictions = svc.predict(test_features)

            for idx, test_prediction in enumerate(test_predictions):

                if idx == 0:
                    scale = 1.5
                elif idx == 1:
                    scale = 2

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    #bbox = ((800, 400), (900, 500))
                    bbox = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                    bboxes.append(bbox)

                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    # record image changes
    if save:
        plt.imshow(draw_img)
        plt.savefig('./output_images/boxed_{}'.format(fname.split('\\')[-1]))

    heat = np.zeros_like(draw_img[:, :, 0]).astype(np.float)

    heat_image = add_heat(heat, bboxes)
    heat_image = apply_threshold(heat_image, threshold=5)
    heat_image = np.clip(heat_image, 0, 255)


    heatmaps.append(heat_image)

    n = 10
    # calculate mean average
    prev_n_heatmaps = heatmaps[-n:]
    avg_heatmap = sum(prev_n_heatmaps) / len(prev_n_heatmaps)

    # calculate weighted average
    weighted_avg_heatmap = np.average(prev_n_heatmaps, axis=0, weights=range(1, len(prev_n_heatmaps) + 1))


    # # get last n coefficients
    # n_coeffs = np.array(heatmaps[-n:])
    # A_l = np.average(n_coeffs[:, 0], weights=range(1, len(n_coeffs) + 1) )

    from scipy.ndimage.measurements import label

    labels = label(weighted_avg_heatmap)

    # sanity checks
    # discard any disproportionate labels

    final_img = draw_labeled_bboxes(original_img, labels)

    if save:
        plt.imshow(final_img)
        plt.savefig('./output_images/final_{}'.format(fname.split('\\')[-1]))

        plt.imshow(heat_image)
        plt.savefig('./output_images/heat_{}'.format(fname.split('\\')[-1]))

    return final_img


    # out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)