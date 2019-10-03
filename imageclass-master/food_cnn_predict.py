from PIL import Image
import imagehash
from os import listdir
from PIL import Image as PImage
import cv2
import os
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import uuid
from keras.preprocessing import image as image_utils
from PIL import Image, ImageTk
import requests
from io import BytesIO
from tkinter import Tk,Label,Canvas,NW,Entry,Button 

from skimage.measure import compare_ssim


def similarity(path):
	imagesList = listdir(path)
	print(imagesList)
	for image in imagesList:
		if image.endswith(".jpg"):
			firstimage = cv2.imread(path+'/'+image)
			print(path+'/'+image)
			break 
	for image in imagesList:
		if image.endswith(".jpg"):
			secimage = cv2.imread(path+'/'+image)
			print(path+'/'+image)
			grayA = cv2.cvtColor(firstimage, cv2.COLOR_BGR2GRAY)
			grayB = cv2.cvtColor(secimage, cv2.COLOR_BGR2GRAY)		
			(score, diff) = compare_ssim(grayA, grayB, full=True)
			diff = (diff * 255).astype("uint8")
			firstimage= secimage
			print("SSIM: {}".format(score))


























#Prediction on a new picture
url = ''
window = Tk()
window.title("Welcome to Image predictor") 
window.geometry('800x600')
lbl = Label(window, text="Enter the URL of the image", font=("Helvetica", 16))
lbl.pack()
def clicked(): 

	#getting the new image from window.
	global url
	lbl.configure()
	url  = (User_input.get())
	print(url)
	response = requests.get(url)
	test_image = Image.open(BytesIO(response.content))
	put_image = test_image.resize((400,400)) 
	#contour Identification
	img_hsv = cv2.cvtColor(np.float32(put_image), cv2.COLOR_BGR2HSV_FULL)
	# Filter out low saturation values, which means gray-scale pixels(majorly in background)
	bgd_mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([255, 30, 255]))
	# Get a mask for pitch black pixel values
	black_pixels_mask = cv2.inRange(np.float32(put_image), np.array([0, 0, 0]), np.array([70, 70, 70]))
	# Get the mask for extreme white pixels.
	white_pixels_mask = cv2.inRange(np.float32(put_image), np.array([230, 230, 230]), np.array([255, 255, 255]))
	final_mask = cv2.max(bgd_mask, black_pixels_mask)
	final_mask = cv2.min(final_mask, ~white_pixels_mask)
	final_mask = ~final_mask
	final_mask = cv2.erode(final_mask, np.ones((3, 3), dtype=np.uint8))
	final_mask = cv2.dilate(final_mask, np.ones((5, 5), dtype=np.uint8))
	# Now you can finally find contours.
	contours, hierarchy = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	final_contours = []
	count=0
	


	#image Selection


	for contour in contours:
		area = cv2.contourArea(contour)
		if area > 14000:
			print("one contour is invalid")
		else:
			final_contours.append(contour)
			count=count+1;
	if (count== 0):
		print("Image is invalid")
	for i in range(len(final_contours)):
		img_bgr = cv2.drawContours(np.float32(put_image), final_contours, -1, (0,255,0), 3)
	debug_img = np.float32(put_image)
	debug_img = cv2.resize(debug_img, None, fx=0.3, fy=0.3)
	cv2.imwrite("./out.png", debug_img)
	test_image = test_image.resize((128,128))  
	img = ImageTk.PhotoImage(put_image)
	pic = Label(image=img)
	pic.pack()
	pic.image = img
	test_image = image_utils.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)



      #load model for class identification  
	img_width, img_height = 128, 128
	model_path = './models/class/model.h5'
	model_weights_path = './models/class/weights.h5'
	model = load_model(model_path)
	model.load_weights(model_weights_path)
	result = model.predict_on_batch(test_image)
	if result[0][0] == 1:

		#load models for food item identification 
		fmodel_path='./models/food/cooked/model.h5'
		fmodel_weights_path = './models/food/cooked/weights.h5'
		fmodel = load_model(model_path)
		fmodel.load_weights(model_weights_path)
		fresult = fmodel.predict_on_batch(test_image)
		if fresult[0][0]==1:

			#assigning new names
			name=str(uuid.uuid4())
			#grouping pics into seperate folders. 
			put_image.save('./groups/pizza/'+name+'.jpg')
			#similarity measure
			similarity('./groups/pizza')
			#final answer on the window
			ans = 'cooked pizza'
			
		else:  
			#repeating the statements of the above case for all upcoming cases.
			name=str(uuid.uuid4())
			put_image.save('./groups/samosa/'+name+'.jpg')	
			similarity('./groups/samosa')
			ans='cooked samosa'

	elif result[0][1] == 1:
		#load models for food item identification 
		fmodel_path = './models/food/fries/model.h5'
		fmodel_weights_path = './models/food/fries/weights.h5'
		fmodel = load_model(model_path)
		fmodel.load_weights(model_weights_path)
		fresult = fmodel.predict_on_batch(test_image)
		if fresult[0][0]==1:
			ans = 'French Fries'
			name=str(uuid.uuid4())			
			put_image.save('./groups/french_fries/'+name+'.jpg')
			similarity('./groups/french_fries')
			
		else:
			name=str(uuid.uuid4())
			put_image.save('./groups/fried_rice/'+name+'.jpg')
			similarity('./groups/fried_rice')
			ans='Fried Rice'
			
	elif result[0][2] == 1:
		#load models for food item identification 
		fmodel_path = './models/food/fruits/model.h5'
		fmodel_weights_path = './models/food/fruits/weights.h5'
		fmodel = load_model(model_path)
		fmodel.load_weights(model_weights_path)
		fresult = fmodel.predict_on_batch(test_image)
		if fresult[0][0]==1:
			name=str(uuid.uuid4())			
			ans = 'Fruits Apple'
			put_image.save('./groups/apple/'+name+'.jpg')
			similarity('./groups/apple')
		else:
			name=str(uuid.uuid4())
			ans='Fruits Banana'
			put_image.save('./groups/banana/'+name+'.jpg')
			similarity('./groups/banana')
				
	out = Label(window, text  = 'Predicted answer : ' +  ans, font=("Helvetica", 16))
	os.system('python object_size.py --image test.png --width 0.955')
	out.pack()


	
	



User_input = Entry(width = 100)
User_input.pack()
btn = Button(window, text="Detect Image", font=("Helvetica", 12), command=clicked)
btn.pack()
window.mainloop()


