TEST_IMAGES_DIR = "./test_images"
SVC_PATH = "./pickle/svc.obj"
X_SCALER_PATH = "./pickle/X_scaler.obj"

HOG = "hog"
SPATIAL = "spatial"
HISTOGRAM = "hist"

params = {
    "ystart": 360,
    "ystop": 720,
    "scales": [1.5, 2.],
    "max_sample_size": None,
    "heat_threshold": 4,
    "smoothing": ("avg", 10),
    "features": {
        HOG: {
            "hog_channel": "ALL",  # options 0, 1, 2, "ALL"
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