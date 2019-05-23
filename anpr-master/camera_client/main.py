"""
        Standard packadges
"""
import subprocess
import time
import cv2
import numpy as np
import requests
import json
import base64
import datetime
"""
        Own Modules
"""
import camera
UNPROCESSED_FOLDER = 'images_unprocessed'
SUCCESS_FOLDER = 'image_success'
CAMERALOCATION = 'PYCONZA2015'
SERVER_LOC = 'http://127.0.0.1:8080'
PUSH_URL = SERVER_LOC +'/pushImage'
cam = camera.ipCamera('http://192.168.1.100/video4.mjpg',None,None)

while True:
    #print("Save file",i)
    cap = cv2.VideoCapture("testData/IMG_0093.mov")
    while not cap.isOpened():
        cap = cv2.VideoCapture("testData/IMG_0093.mov")
        cv2.waitKey(1000)
        print "Wait for the header"

    pos_frame = cap.get(1)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            cv2.imshow('video', frame)
            pos_frame = cap.get(1)
            print str(pos_frame)+" frames"
#     cv2.imshow('Test Image',frame)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()
            t0 = time.time()
            imagename = str(CAMERALOCATION+str(datetime.datetime.now()).replace(" ","_").replace(":","_"))+'.jpeg'
            imagelocation = 'images_unprocessed/'+imagename
            cv2.imwrite(imagelocation, frame)
    
            p = subprocess.Popen(["python", "nanpr.py","-i",imagelocation],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
            output = p.communicate()[0]
            print("Number Plate Found:",output)
    #    if len(output) > 3:
    #        r = requests.post(PUSH_URL,data={'data':json.dumps({'numberplate':output,'camlocation':CAMERALOCATION,'jsondata': base64.b64encode(open(imagelocation, 'rb').read())})})
    #        print("Time to process: ",time.time()-t0)
            time.sleep(15)
        #python adaptive_thresholding.py -i test_data/2382709.jpg

        else:
            # The next frame is not ready, so we try to read it again
            cap.set(1, pos_frame-1)
            print "frame is not ready"
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(1) == cap.get(7):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break



