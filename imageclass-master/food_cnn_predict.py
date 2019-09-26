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
#load model
img_width, img_height = 128, 128
model_path = './models/class/model.h5'
model_weights_path = './models/class/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
#Prediction on a new picture
from keras.preprocessing import image as image_utils
from PIL import Image, ImageTk
import requests
from io import BytesIO
from tkinter import Tk,Label,Canvas,NW,Entry,Button 
url = ''
window = Tk()
window.title("Welcome to Image predictor") 
window.geometry('800x600')
lbl = Label(window, text="Enter the URL of the image", font=("Helvetica", 16))
lbl.pack()
def clicked(): 
	global url
	lbl.configure()
	url  = (User_input.get())
	print(url)
	response = requests.get(url)
	test_image = Image.open(BytesIO(response.content))
	put_image = test_image.resize((400,400)) 
	put_image.save('./test1.png')
	test_image = test_image.resize((128,128))  
	img = ImageTk.PhotoImage(put_image)
	pic = Label(image=img)
	pic.pack()
	pic.image = img
	test_image = image_utils.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	result = model.predict_on_batch(test_image)
	if result[0][0] == 1:
		fmodel_path='./models/food/cooked/model.h5'
		fmodel_weights_path = './models/food/cooked/weights.h5'
		fmodel = load_model(model_path)
		fmodel.load_weights(model_weights_path)
		fresult = fmodel.predict_on_batch(test_image)
		if fresult[0][0]==1:
			name=str(uuid.uuid4())
			put_image.save('./groups/pizza/'+name+'.jpg')
			ans = 'cooked pizza'
			
		else:
			ans='cooked samosa'
			name=str(uuid.uuid4())
			put_image.save('./groups/samosa/'+name+'.jpg')
			
	elif result[0][1] == 1:
		fmodel_path = './models/food/fries/model.h5'
		fmodel_weights_path = './models/food/fries/weights.h5'
		fmodel = load_model(model_path)
		fmodel.load_weights(model_weights_path)
		fresult = fmodel.predict_on_batch(test_image)
		if fresult[0][0]==1:
			ans = 'French Fries'
			name=str(uuid.uuid4())			
			put_image.save('./groups/french_fries/'+name+'.jpg')
			
		else:
			name=str(uuid.uuid4())
			put_image.save('./groups/fried_rice/'+name+'.jpg')
			ans='Fried Rice'
			
	elif result[0][2] == 1:
		fmodel_path = './models/food/fruits/model.h5'
		fmodel_weights_path = './models/food/fruits/weights.h5'
		fmodel = load_model(model_path)
		fmodel.load_weights(model_weights_path)
		fresult = fmodel.predict_on_batch(test_image)
		if fresult[0][0]==1:
			name=str(uuid.uuid4())			
			ans = 'Fruits Apple'
			put_image.save('./groups/apple/'+name+'.jpg')
			
		else:
			name=str(uuid.uuid4())
			ans='Fruits Banana'
			put_image.save('./groups/banana/'+name+'.jpg')
				
	out = Label(window, text  = 'Predicted answer : ' +  ans, font=("Helvetica", 16))
	os.system('python object_size.py --image test.png --width 0.955')
	out.pack()

def similarity(path):
	imagesList = listdir(path)
	loadedImages = []
	for image in imagesList:
		h_img = PImage.open(path + image)
		loadedImages.append(h_img)
		one_hash = imagehash.average_hash(h_img)
		break 
	for image in imagesList:
		h_img = PImage.open(path + image)
		loadedImages.append(h_img)
		two_hash = imagehash.average_hash(h_img)
		print(one_hash - two_hash)
		one_hash = two_hash
	
	
	



User_input = Entry(width = 100)
User_input.pack()
btn = Button(window, text="Detect Image", font=("Helvetica", 12), command=clicked)
btn.pack()
window.mainloop()


