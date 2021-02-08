import cv2
import random
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Capture camera 0 - default camera
webcam = cv2.VideoCapture(0)

while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()
    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for i in range(len(face_coordinates)):
        (x, y, w, h) = face_coordinates[i]
        # print(len(face_coordinates))

        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (random.randint(128, 256), random.randint(128, 256), random.randint(128, 256)), 3)
    cv2.imshow("Webcam",frame)
    cv2.waitKey(1)


print("Code Completed")
