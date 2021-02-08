import cv2
import random

#Load some pre-trained data on face frontals from opencv(haar cascade algorithm)
#Classifier can detect face
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in
img = cv2.imread('people.jpg')
#Resize the image
img = cv2.resize(img,(1280,640))

#Change it to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Print face coordinates
#print(face_coordinates)

#Tuple for rectangle points
for i in range(len(face_coordinates)):
    (x,y,w,h) = face_coordinates[i]
    #print(len(face_coordinates))

    cv2.rectangle(img,(x, y),(x+w, y+h), (random.randint(128,256),random.randint(128,256),random.randint(128,256)), 3)
#Draw rectangles around the faces


#Display image
cv2.imshow("Image",img)

#Pause execution
cv2.waitKey(0)



print("Code completed")
