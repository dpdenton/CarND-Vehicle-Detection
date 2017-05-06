import pickle
from utils import *
from conts import *
heatmaps = []

def process_image(img, fname="video.jpg", save=False):

    print("Using {}".format(params))

    # load trained svc and scaler
    with open(SVC_PATH, "rb") as f:
        svc = pickle.load(f)
    with open(X_SCALER_PATH, "rb") as f:
        X_scaler = pickle.load(f)

    bboxes = []

    # get params
    ystart = params["ystart"]
    ystop = params["ystop"]
    orient = params["features"][HOG]["orient"]
    pix_per_cell = params["features"][HOG]["pix_per_cell"]
    cell_per_block = params["features"][HOG]["cell_per_block"]
    cspace = params["features"][HOG]["cspace"]
    smoothing_method = params["smoothing"]

    draw_img = np.copy(img)
    original_img = np.copy(img)

    # convert colour space and normalize (glob is already between 0-1)
    img = convert_color(img, cspace)
    img = img.astype(np.float64) / 255

    ctrans_tosearch = img[ystart:ystop, :, :]

    imshape = ctrans_tosearch.shape

    # set scaled hog features
    scale_dict = {}

    for scale in params['scales']:
        scale_dict[scale] = {}
        scale_dict[scale]['resize'] = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    for scale in scale_dict:

        scale_dict[scale]['ch1'] = scale_dict[scale]['resize'][:, :, 0]
        scale_dict[scale]['ch2'] = scale_dict[scale]['resize'][:, :, 1]
        scale_dict[scale]['ch3'] = scale_dict[scale]['resize'][:, :, 2]

    # Define blocks and steps as above
    nxblocks = (scale_dict[1.5]['resize'].shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (scale_dict[1.5]['resize'].shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    for scale in scale_dict:

        # Compute individual channel HOG features for the entire image
        scale_dict[scale]['hog1'] = get_hog_features(scale_dict[scale]['ch1'], orient, pix_per_cell, cell_per_block, feature_vec=False)
        scale_dict[scale]['hog2'] = get_hog_features(scale_dict[scale]['ch2'], orient, pix_per_cell, cell_per_block, feature_vec=False)
        scale_dict[scale]['hog3'] = get_hog_features(scale_dict[scale]['ch2'], orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):

            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # init feature arrays
            features = np.empty((0))
            init_features = True

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            if HOG in params['features']:

                for idx, scale in enumerate(scale_dict):

                    scale_dict[scale]['hog_feat1'] = scale_dict[scale]['hog1'][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    scale_dict[scale]['hog_feat2'] = scale_dict[scale]['hog2'][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    scale_dict[scale]['hog_feat3'] = scale_dict[scale]['hog3'][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                    hog_features = np.hstack((scale_dict[scale]['hog_feat1'], scale_dict[scale]['hog_feat2'], scale_dict[scale]['hog_feat3']))

                    if init_features:
                        features = np.hstack((features, scale_dict[scale]['hog_feat1'], scale_dict[scale]['hog_feat2'], scale_dict[scale]['hog_feat3']))
                        init_features = False
                    else:
                        hog_features = np.hstack((scale_dict[scale]['hog_feat1'], scale_dict[scale]['hog_feat2'], scale_dict[scale]['hog_feat3']))
                        if features.shape == hog_features.shape:
                            features = np.vstack((features, hog_features))

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
            test_predictions = svc.predict(test_features)

            for idx, test_prediction in enumerate(test_predictions):

                # set scale
                scale = params['scales'][idx]

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    #bbox = ((800, 400), (900, 500))
                    bbox = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                    bboxes.append(bbox)

                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    # set heatmaps
    heat = np.zeros_like(draw_img[:, :, 0]).astype(np.float)
    heat_image = add_heat(heat, bboxes)
    heat_image = apply_threshold(heat_image, threshold=params['heat_threshold'])
    heat_image = np.clip(heat_image, 0, 255)
    heatmaps.append(heat_image)

    # calculate mean average
    if smoothing_method[0] == 'avg':

        n = smoothing_method[1]
        prev_n_heatmaps = heatmaps[-n:]
        heatmap = sum(prev_n_heatmaps) / len(prev_n_heatmaps)

    # calculate weighted average
    elif smoothing_method[0] == 'weighted':

        n = smoothing_method[1]
        prev_n_heatmaps = heatmaps[-n:]
        heatmap = np.average(prev_n_heatmaps, axis=0, weights=range(1, len(prev_n_heatmaps) + 1))

    # use last map
    else:
        heatmap = heatmaps[-1]

    # find labels and draw onto original image
    from scipy.ndimage.measurements import label
    labels = label(heatmap)
    final_img = draw_labeled_bboxes(original_img, labels)

    # record image changes
    if save:

        plt.imshow(draw_img)
        plt.savefig('./output_images/boxed_{}'.format(fname.split('\\')[-1]))

        plt.imshow(final_img)
        plt.savefig('./output_images/final_{}'.format(fname.split('\\')[-1]))

        plt.imshow(heat_image)
        plt.savefig('./output_images/heat_{}'.format(fname.split('\\')[-1]))

        plt.cla()

    return final_img


    # out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)