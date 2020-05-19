import cv2
import numpy as np
import os
import requests
import time
import boto3
from botocore.exceptions import ClientError
import random
import string
import requests

ACCESS_KEY = 'AKIAI5MIUR3A3SS3A7TQ'
SECRET_KEY = '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'


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Matan']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

def randomString(stringLength=8):
     letters = string.ascii_lowercase
     return ''.join(random.choice(letters) for i in range(stringLength))

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)

    except ClientError as e:
        logging.error(e)
        return False
    return True

while True:
    ret, img =cam.read()
    img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])





        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 70):
            id = names[id]
            confidencePct = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidencePct = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidencePct), (x+5,y+h-5), font, 1, (255,255,0), 1)

        if (confidence < 70):
            imageName = "detection/User." + str(id) + '.' + randomString(10) + ".jpg"

            cv2.imwrite(imageName, img)

            entryName = "entries/" + randomString(64) + ".jpg"

            response = upload_file(imageName, "tikal-rpi", entryName)

            d = {'user_id': id, 'name': 3, 'image_key': entryName}

            requests.post("https://36o0y7kjle.execute-api.us-east-1.amazonaws.com/dev/register", data=d)

            print("https://tikal-rpi.s3-eu-west-1.amazonaws.com/{0}".format(entryName))

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
