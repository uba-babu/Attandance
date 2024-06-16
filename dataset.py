import cv2
import pickle
import numpy as np
import os

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Load the Haar cascade for face detection
facedetect = cv2.CascadeClassifier(r'C:\Users\U.B.A Yadav\OneDrive\Desktop\attendance\haarcascade_frontalface_default .xml')  # Ensure the path is correct

# Initialize video capture from webcam
video = cv2.VideoCapture(0)

faces_data = []
i = 0

# Get the name of the person to label the data
name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert face data to numpy array and reshape
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Load existing names and faces data if they exist
names_file = 'data/names.pkl'
faces_file = 'data/faces_data.pkl'

if os.path.exists(names_file):
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
else:
    names = []

names += [name] * 100

with open(names_file, 'wb') as f:
    pickle.dump(names, f)

if os.path.exists(faces_file):
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
else:
    faces = faces_data

with open(faces_file, 'wb') as f:
    pickle.dump(faces, f)

print("Training data saved successfully!")
