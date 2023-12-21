import cv2
from utils import *
from darknet import Darknet


m=Darknet("yolov3.cfg")
m.load_weights("yolov3.weights")
classes= load_class_names("coco.names")
obj= cv2.imread("images/room.jpg")
obj= cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
img = cv2.resize(obj, (m.width, m.height))
boxes = detect_objects(m, img,  0.4, 0.6)
plot_boxes(obj, boxes, classes, plot_labels=True)