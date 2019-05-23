
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract
import argparse
import glob
import math
import itertools
import operator
import imutils

def preprocess(img):
	config = {"y_offset": 20, # maximum y offset between chars
		    "x_offset":  55, # maximum x gap between chars
		    "thesh_offset":  0, # this determines the cutoff point on the adaptive threshold.
		    "thesh_window": 25, # window of adaptive theshold area
		    # max min char width, height and ratio
		    "w_min":  6, # char pixel width min
		    "w_max":  30, # char pixel width max
		    "h_min":  12, # char pixel height min
		    "h_max":  40, # char pixel height max
		    "hw_min":  1.5, # height to width ration min
		    "hw_max":  3.5, # height to width ration max
		    "h_ave_diff":  1.09,  # acceptable limit for variation between characters
	}
	#cv2.imshow("Input",img)
	image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, config["thesh_window"], config["thesh_offset"])
	allblobs = img.copy()
	reducedblobs = img.copy()
	roiblobs = img.copy()
	plateBlob = img.copy()

	im2, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	correctly_sized_list = []
	for c in cnts:
		#print c
		points = cv2.boundingRect(c)
		x = points[0]
		y = points[1]
		w = points[2]
		h = points[3]
		#if self.debug:
		cv2.rectangle(allblobs, (x, y), (x + w, y + h), (0, 255, 0), 2) 
	    # Filter on width and height
		if config["w_min"] < w < config["w_max"] and config["h_min"] < h < config["h_max"] and config["hw_min"] < 1.0*h/w < config["hw_max"]:
			correctly_sized_list.append((x, y, w, h))
			cv2.rectangle(reducedblobs, (x, y), (x + w, y + h), (0, 255, 0), 2) 

	# now we try to filter based on character proximity and the fact that they would be in a row
	possible_plate_regions = []
	# sort by x position
	sort_list = sorted(correctly_sized_list, key=lambda x: x[0])

	# Try to group blobs into platelike groups
	for char in sort_list:
		placed_char = False
	    # Check if this blob has same y and x within offset values off current are (this is why we sorted by x value).
		for region in possible_plate_regions:
			if region[-1][1] - config["y_offset"] < char[1] < region[-1][1] + config["y_offset"] and region[-1][0] + config["x_offset"] > char[0]:
				region.append(char)
				placed_char = True
				break
	    # if char was not placed in a group, it becomes the first of a new group
		if placed_char is False:
			possible_plate_regions.append([char])

	# Now remove chars from regions if heights differ significantly, as numberplate chars are evenly sized. This could possibly be done in above filter, but this seemed better
	possible_plate_regions_ave_filtered = []

	for region in possible_plate_regions:
		if len(region)>2:
			possible_plate_regions_ave_filtered.append([])
			ave = sum([char[3] for char in region])/len(region)
			for char in region:
				if ave/config["h_ave_diff"] < char[3] < ave*config["h_ave_diff"]:
					possible_plate_regions_ave_filtered[-1].append(char)

	possible_plate_regions_ave_filtered = [x for x in possible_plate_regions_ave_filtered if len(x)>2]

	possible_plate_regions_plate_details = []
	
	for region in possible_plate_regions_ave_filtered:
	    # Find the min and max values of the plate region
		xmin = min([x[0] for x in region])
		ymin = min([x[1] for x in region])
		xmax = max([x[0]+x[2] for x in region])
		ymax = max([x[1]+x[3] for x in region])
		topleft = sorted(region, key=lambda x: x[0]+x[1])[0]
		topright = sorted(region, key=lambda x: -(x[0]+x[2])+x[1])[0]
		botleft = sorted(region, key=lambda x: x[0]-(x[1]+x[3]))[0]
		botright = sorted(region, key=lambda x: -(x[0]+x[2])-(x[1]+x[3]))[0]
	    
	    #print (topleft, topright, botleft, botright)
	    
		mtop = 1.0*(topleft[1]-topright[1])/(topleft[0]-(topright[0]+topright[2]))
		mbot = 1.0*(botleft[1]+botleft[3]-(botright[1]+botright[3]))/(botleft[0]-(botright[0]+botright[2]))
	    #print mtop, mbot

		cv2.rectangle(plateBlob, (xmin - 10, ymin - 10),(xmax + 20, ymax + 20), (0,0,255), 1) 
		
		#if "true" == "true" :
			#for char in region:
				#(x, y, w, h) = char
				#cv2.rectangle(roiblobs, (x, y), (x + w, y + h), (0, 0, 255), 1)

				
	    
		possible_plate_regions_plate_details.append({"size": (xmin, ymin, xmax, ymax), 
							 "roi": (xmin - 2*config["w_max"], ymin - config["h_max"], xmax + 2*config["w_max"], ymax + config["h_max"]),
							 "average_angle": (mtop + mbot)/2.0})
	    # Get area plus 2 x max char width to the sides and max char height above and below
##              try:
##                      skew_correct(possible_plate_regions_plate_details[-1])
##              
##              # use thresholded roi to find chars again
##                      if "warped2" in possible_plate_regions_plate_details[-1] and possible_plate_regions_plate_details[-1]["warped2"] is not None:
##                              detect_chars(possible_plate_regions_plate_details[-1])
##                              if len(possible_plate_regions_plate_details[-1]["plate"])>3:
##                                      possible_plate_regions_plate_details[-1]["somechars"] = True
##              except Exception as ex:
##                      print ex
	    
	
	#cv2.imshow("Blobs ALL", allblobs)
	#cv2.imshow("Blobs size filter", reducedblobs)
	#cv2.imshow("Blobs group filtered", roiblobs)
	cv2.imshow("Plate", plateBlob)
	#text = pytesseract.image_to_string(plate_im)
	#print(text)
        #plates = posible_plate_regions_plate_details
	#return plates



if __name__ == '__main__':
	print("DETECTING PLATE . . .")

	cap = cv2.VideoCapture("testData/IMG_0093.mov")
	while not cap.isOpened():
		cap = cv2.VideoCapture("testData/IMG_0096 2.mov")
		cv2.waitKey(1000)
		print ("Wait for the header")

	pos_frame = cap.get(1)
	while True:
		flag, frame = cap.read()
		if flag:
	       # The frame is ready and already captured
			frame = imutils.resize(frame, width=1000)
			#cv2.imshow('video', frame)
			pos_frame = cap.get(1)
			#print str(pos_frame)+" frames"
			preprocess(frame)
		else:
	       # The next frame is not ready, so we try to read it again
			cap.set(1, pos_frame-1)
			print ("frame is not ready")
	       # It is better to wait for a while for the next frame to be ready
			cv2.waitKey(1000)
	
		if cv2.waitKey(10) == 27:
			break
		if cap.get(1) == cap.get(7):
	       # If the number of captured frames is equal to the total number of frames,
	       # we stop
		       break

##      #img = cv2.imread("testData/Final.JPG")
##      img = cv2.imread("testData/test3.jpg")
##
##      
##      #print(contours)
##      #if len(contours)!=0:
##      #       print (len(contours)) #Test
##      #       cv2.drawContours(img,contours,-1,(0,255,0),1)
##      #       cv2.imshow("Contours",img)
##      #       cv2.waitKey(0)
##
##
##      cleanAndRead(img,contours)
