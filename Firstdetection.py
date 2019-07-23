from imageai.Detection import ObjectDetection
import os
import numpy as np
import matplotlib.pyplot

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
#print(detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg")))
ct=0
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
   # print(eachObject["percentage_probability"])




#print(list)
