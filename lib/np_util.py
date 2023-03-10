import numpy as np
import cv2
from types import SimpleNamespace as SNS
from skimage import feature as skFeat


def crop(img, top=0, btm=0, left=0, right=0):
    if hasattr(img, 'crop') and callable(getattr(img, 'crop')):
        # img is a PIL object
        w, h = img.size[:2]
        return img.crop((left, top, w - 10, h - 10))

    elif hasattr(img, 'shape'):
        # img is a numpy array, eg. via cv2.imread()
        h, w = img.shape[:2]
        return img[top:h - btm, left:w - right]

    else:
        raise ValueError('img type ' + type(img) + ' not expect')


def BGRto(cs, img):
    # matplotlib.image.imread returns RGB
    if cs == 'RGB': return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cs == 'LUV': return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if cs == 'YUV': return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if cs == 'HSV': return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if cs == 'HLS': return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if cs == 'YCrCb': return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img


def RGBto(cs, img):
    # cv2.imread returns BGR
    if cs == 'BGR': return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if cs == 'LUV': return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if cs == 'YUV': return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if cs == 'HSV': return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if cs == 'HLS': return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if cs == 'YCrCb': return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return img


def colorSpaceRanges(cs):
    return [(0, 192), (0, 256), (0, 256)] if cs == 'LUV' else [(0, 256)] * 3


def spatial_features(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, bins=32, ranges=[(0, 256)] * 3):
    ch1 = np.histogram(img[:, :, 0], bins=bins, range=ranges[0])
    ch2 = np.histogram(img[:, :, 1], bins=bins, range=ranges[1])
    ch3 = np.histogram(img[:, :, 2], bins=bins, range=ranges[2])
    chs = [ch1, ch2, ch3]
    return SNS(
        hist=np.concatenate([ch[0] for ch in chs]),
        bin_edges=np.concatenate([ch[1] for ch in chs]),
    )


def hog_vis(img,
            orientations=8,
            pxs_per_cell=8,
            cells_per_blk=2,
            feature_vector=False):
    ''' Histogram of Oriented Gradients, visualise=True 
    '''
    result = skFeat.hog(img,
                        orientations=orientations,
                        pixels_per_cell=(pxs_per_cell, pxs_per_cell),
                        cells_per_block=(cells_per_blk, cells_per_blk),
                        transform_sqrt=True, visualise=True, feature_vector=feature_vector
                        )
    return SNS(
        features=result[0],
        images=result[1],
    )


def image_features(img,
                   spatial_size=None,  # to add spatial feature, pass eg. (32,32)
                   hist_bins=0,  # to add histogram feature, pass number of bins
                   hist_ranges=None,  # or pass ranges
                   hog_params=None,  # to add hog feature, pass params dict for hog()
                   ):
    features = []
    if spatial_size:
        features.append(spatial_features(img, spatial_size))
    if hist_bins or hist_ranges:
        features.append(color_hist(img, hist_bins, hist_ranges).hist)
    if hog_params:
        features.append(hog(img,
                            hog_params.get('orientations', 8),
                            hog_params.get('pxs_per_cell', 8),
                            hog_params.get('cells_per_blk', 2),
                            hog_params.get('channels', 'all'),
                            hog_params.get('visualise', False),
                            hog_params.get('feature_vector', False)))
    return np.concatenate(features)


def images_features(imgspath,
                    color_space='',  # color space to convert to, eg. 'YCrCb'
                    spatial_size=None,  # to add spatial feature, pass eg. (32,32)
                    hist_bins=0,  # to add histogram feature, pass number of bins
                    hist_ranges=None,  # or pass ranges
                    hog_params=None,  # to add hog feature, pass params dict for hog()
                    ):
    ret = []
    for imgpath in imgspath:
        img = BGRto(color_space, cv2.imread(imgpath))
        ret.append(image_features(img, spatial_size, hist_bins, hist_ranges, hog_params))
    return ret
