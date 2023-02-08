from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2, string
from lib import np_util as npu
from lib import feature_extraction as fe
from lib import draw
from lib.helpers import _x0, _x1, _y0, _y1, _wd, _ht


def find_hot_boxes(img, box_rows, model, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True, hog_feats=None):
    ''' Returns list of hot windows - windows which prediction is positive for
        box in windows.
    model: object with .classifier and .scaler
    '''
    result = []
    img = npu.RGBto(color_space, img)
    wins_cnt = 0

    if hog_feat:
        # Tried to crop with img[360:] "ValueError: operands could not be broadcast together with shapes...",
        hog_img_feats = fe.hog_features(img, orient, pix_per_cell, cell_per_block,
                                        hog_channel, feature_vec=False)

    print(img.shape)
    for row in box_rows:
        for box in row:
            b = X0Y0(box)
            # print("%d %d : %d %d" % (b.y0, b.y1, b.x0, b.x1))
            im = img[b.y0:b.y1, b.x0:b.x1]
            # cv2.imshow("%d %d : %d %d" % (b.y0, b.y1, b.x0, b.x1), im)
            # cv2.waitKey(0)
            test_img = cv2.resize(im, model.train_size)
            if hog_feat:
                x0 = b.x0 // pix_per_cell
                y0 = b.y0 // pix_per_cell  # - 360 if hog_roi_feats works
                wd = test_img.shape[1] // pix_per_cell - (cell_per_block - 1)
                hog_feats = hog_img_feats[:, y0:y0 + wd, x0:x0 + wd].ravel()

            features = fe.image_features(test_img, color_space=None,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel,
                                         spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feats=hog_feats,
                                         concat=True)

            test_features = model.scaler.transform(features.reshape(1, -1))
            prediction = model.classifier.predict(test_features)
            # print(prediction)
            if prediction == 1:
                result.append(box)
    return result


def compute_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
    boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)

    # compute the IOU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


class X0Y0():
    def __init__(self, bbox):
        ''' bbox of ((x0,y0),(x1,y1))
        '''
        self.x0 = _x0(bbox)
        self.x1 = _x1(bbox)
        self.y0 = _y0(bbox)
        self.y1 = _y1(bbox)
        self.wd = self.x1 - self.x0


car_labels = string.ascii_uppercase[:26]


class Car():
    def __init__(self, b, ilabel):
        ''' b: ((x0,y0),(x1,y1)) '''
        self.wins = b  # detected windows of heat
        self.nwins = 1  # number of windows per frame
        self.label = car_labels[ilabel % len(car_labels)]

    def overlaps(self, b):
        iou = compute_IOU(self.wins, b)
        if iou > 0.4:
            tmp = ((min(_x0(self.wins), _x0(b)), min(_y0(self.wins), _y0(b))),
                         (max(_x1(self.wins), _x1(b)), max(_y1(self.wins), _y1(b))))
            if _ht(tmp) <= _wd(tmp):
                self.wins = tmp
            else:
                self.wins = b if _wd(b) * _ht(b) < _wd(self.wins) * _ht(self.wins) else self.wins
            return True

        elif 0.2 < iou <= 0.4:
            self.wins = b if _wd(b) * _ht(b) < _wd(self.wins) * _ht(self.wins) else self.wins
            return True

        else:
            return False


class CarDetector():
    def __init__(self, model, img_shape):
        self.rows = None
        self.cars = []
        self.model = model
        self.icar = -1
        self.iheat = -1

    def find_hot_wins(self, img):
        ''' Returns bounding windows of heats in img
            Updates self.dbg_wins and self.dbg_txts
        '''
        defaults = self.model.defaults
        if self.rows is None:
            self.rows = fe.bbox_rows(img.shape, ymin=img.shape[0] // 2, max_h=512)

        hot_boxes = find_hot_boxes(img, self.rows, self.model, **defaults)
        print("Num hotbox: %d" % len(hot_boxes))

        return hot_boxes

    @property
    def next_icar(self):
        ''' Increments self.icar and return it
        '''
        self.icar += 1
        return self.icar

    def detect(self, img):
        ''' Detect cars in img and add them to self.cars.
            Updates self.dbg_wins and self.dbg_txts
        '''

        dbg_img = np.copy(img)
        new_wins = self.find_hot_wins(img)

        new_cars = []
        added_to = []
        bad_size = []
        if new_wins:
            # For those not removed, try grouping them into as least number of new_cars
            for win in new_wins:

                found = True
                for newcar in new_cars:
                    # loop won't run on first iteration as new_cars is empty.
                    if newcar.overlaps(win):  # overlap_by() not needed as new_cars are x0 ordered
                        newcar.nwins += 1
                        if newcar.label not in added_to:
                            added_to.append(newcar.label)
                        found = False
                        break
                if found:
                    new_car = Car(win, self.next_icar)
                    new_cars.append(new_car)

            self.cars.extend(new_cars)

        result = [new_cars[0].wins]
        for new_car in new_cars:
            isNotOverlap = True
            if new_car.nwins > 2:
                for i in range(len(result)):
                    if compute_IOU(result[i], new_car.wins) > 0:
                        tmp = (
                                (min(_x0(result[i]), _x0(new_car.wins)), min(_y0(result[i]), _y0(new_car.wins))),
                                (max(_x1(result[i]), _x1(new_car.wins)), max(_y1(result[i]), _y1(new_car.wins))))
                        if _ht(tmp) <= _wd(tmp):
                            result[i] = tmp
                        else:
                            result[i] = result[i] if _ht(result[i]) * _wd(result[i]) < _ht(new_car.wins) * _wd(
                                new_car.wins) else new_car.wins
                        isNotOverlap = False
                        break

                if isNotOverlap:
                    result.append(new_car.wins)
        for res in result:
            dbg_img = draw.rect(dbg_img, res, (0, 255, 0))

        return dbg_img

    def detected_image(self, img):
        ''' API for running detect(), purge(), and final_purge_and_detection_image()
        '''
        return self.detect(img)
