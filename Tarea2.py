#!/usr/bin/env python

#Required libraries
import numpy as np
import cv2
import pymeanshift as pms
import json

#Variables initialization
cap = cv2.VideoCapture(0)

printMenu = False
mySegmenter = pms.Segmenter()
mkeyPress = False
wKeyPress = False
tkeyPress = False

#Debugging purposes
#params = {}
#params['ms'] = []
#params['ms'].append({
#    'spatial_radius': mySegmenter.spatial_radius,
#    'range_radius': mySegmenter.range_radius,
#    'min_density': mySegmenter.min_density
#})

#with open('params.txt', 'w') as outfile:
#    json.dump(params, outfile)

#Read algorithms parameters from json file
with open('params.txt', 'r') as json_file:
    params = json.load(json_file)

for ms in params['ms']:
    mySegmenter.spatial_radius = ms['spatial_radius']
    mySegmenter.range_radius = ms['range_radius']
    mySegmenter.min_density = ms['min_density']

while(True):

    if not printMenu:
        print("This program performs image segmentation of the images taken with the webcam.")
        print("Press ms in image window to execute image segmentation using mean shift algorithm.")
        print("Press ws in image window to execute image segmentation using watersheds algorithm.")
        print("Press th in image window to execute image segmentation using Otsu thresholding algorithm.")
        print("Press q in image window for exit.")
        printMenu = True

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Tarea2: Segmentacion de imagen',frame)

    key = cv2.waitKey(2)

    if key & 0xFF == ord('q'):
        break

    elif key & 0xFF == ord('m'):
        mKeyPress = True
        wKeyPress = False
        tKeyPress = False

    elif key & 0xFF == ord('w'):
        wKeyPress = True
        mKeyPress = False
        tKeyPress = False

    elif key & 0xFF == ord('t'):
        tKeyPress = True
        mKeyPress = False
        wKeyPress = False

    elif key & 0xFF == ord('s') and mKeyPress:
        print("Using Mean Shift Algorithm!")
        print("Takes around 15s!")

        (segmentedImage, labelsImage, numberRegions) = mySegmenter(frame)

        cv2.imshow('Tarea2: Imagen Segmentada por Mean Shift', segmentedImage)

        print("Number of regions: " + str(numberRegions))
        print("Mask:")
        print(str(len(labelsImage)) + "x" + str(len(labelsImage[0])))
        print(labelsImage)

        mKeyPress = False
        cv2.imwrite('msCamSeg.png', segmentedImage)

    elif key & 0xFF == ord('s') and wKeyPress:
        print("Using Watersheds Algorithm!")

        #Binary image using otsu thresholding
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        #Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

        #Sure background area
        sureBg = cv2.dilate(opening, kernel, iterations = 3)

        #Finding sure foreground area
        distTransform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sureFg = cv2.threshold(distTransform, 0.7 * distTransform.max(), 255, 0)

        # Finding unknown region
        sureFg = np.uint8(sureFg)
        unknown = cv2.subtract(sureBg, sureFg)

        #Marker labelling
        ret, markers = cv2.connectedComponents(sureFg)

        #Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        #Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        #Apply watershed
        markers = cv2.watershed(frame, markers)
        frame[markers == -1] = [255,0,0]

        cv2.imshow('Tarea2: Imagen Segmentada por Watersheds', frame)

        cv2.imwrite('wsCamSeg.png', frame)

        wKeyPress = False

    elif key & 0xFF == ord('h') and tKeyPress:

        print("Using Otsu Thresholding Algorithm!")
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("Threshold: " + str(ret))

        cv2.imshow('Tarea2: Imagen Segmentada por Otsu Thresholding', thresh)

        cv2.imwrite('othCamSeg.png', thresh)

        tKeyPress = False
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
