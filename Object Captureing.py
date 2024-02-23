









import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import xlsxwriter


workbook = xlsxwriter.Workbook('D:\\pyhton project\\Example2.xlsx')
worksheet = workbook.add_worksheet()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
a=[]
row = 0
column = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    x=results.pandas().xyxy[0]
    for i in x['name']:
        a.append(i)
    for i in a:
        worksheet.write(row, column, i)
        row += 1
  
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()