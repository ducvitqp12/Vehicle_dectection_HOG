import numpy as np
import cv2, time
import joblib

from image_detection import *
from config import pklpath, default, defaults
from types import SimpleNamespace as SNS

t = time.time()
model = SNS(
    classifier=joblib.load(pklpath.svc),
    scaler=joblib.load(pklpath.scaler),
    train_size=default.train_size,
    defaults=defaults,
)
detector = CarDetector(model, (720, 1280))


def process_image(img):
    return detector.detected_image(img)


image = cv2.imread('./test_images/test11.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1280, 720))
# print(image)
processed = process_image(image)
processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
cv2.imwrite('output_images/processed.jpg', processed)
