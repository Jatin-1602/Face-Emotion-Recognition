import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                4: "Neutral", 5: "Sad", 6: "Surprised"}


# UPLOAD FUNCTION
def upload(image_file):
    frame = cv2.imread(image_file)
    print(frame.shape)

    face_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_casc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), thickness=2)
        roi_gray = gray[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        print(prediction)
        maxindex = int(np.argmax(prediction))
        print(emotion_dict[maxindex])
        cv2.putText(frame, emotion_dict[maxindex], (x + 30, y+220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Image', cv2.resize(frame, (frame.shape[1], frame.shape[0])))
    cv2.waitKey(0)


# MAIN
path = "Image Path"

if path:
    upload(path)
else:
    print("Image not Uploaded")

