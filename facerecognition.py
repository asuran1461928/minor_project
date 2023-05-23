import cv2
import pickle
import os
import cvzone
import face_recognition
from datetime import datetime
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://attendance-36fe5-default-rtdb.firebaseio.com/",
    'storageBucket':"attendance-36fe5.appspot.com"
})

bucket=storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 1280) #width
cap.set(4, 720) #height

imgBackground = cv2.imread('resources/background.png')
imgBackground = cv2.resize(imgBackground, (1490, 1065)) # resize background image to match resized_img dimensions

folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

#load the encoding file
print("loading encoding file....")
file=open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
#print(studentIds)
print("loaded encoded file")

modeType=0
counter=0
id=-1
imgStudent=[]

while True:
    success , img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    resized_img = cv2.resize(img, (1625, 1625))
    resized_img = cv2.resize(resized_img, (799, 930))  # reduce size further to match imgBackground
    imgBackground[99:99 + 930, 37:37 + 799] = resized_img

    # resize imgModeList[0] to match target region
    imgMode = cv2.resize(imgModeList[modeType], (610, 988))
    imgBackground[38:38 + 988, 858:858 + 610] = imgMode

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
            matches=face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis=face_recognition.face_distance(encodeListKnown, encodeFace)
            #print("matches",matches)
            #print("distance",faceDis)

            matchIndex=np.argmin(faceDis)
            #print("matchIndex",matchIndex)

            if matches[matchIndex]:
                print("known face is detected")
                print(studentIds[matchIndex])
                y1,x2,y2,x1=faceLoc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                bbox = 37+x1, 99+y1, x2-x1, y2-y1
                imgBackground=cvzone.cornerRect(imgBackground,bbox,rt=0)
                id=studentIds[matchIndex]
                print(id)
                if counter==0:
                    cvzone.putTextRect(imgBackground, "Loading", (150, 150))
                    cv2.imshow("face attendance",imgBackground)
                    cv2.waitKey(1)
                    counter=1
                    modeType=2

        if counter!=0:

            if counter==1:
                #get the data
                studentinfo=db.reference(f'Students/{id}').get()
                print(studentinfo)
                #get the image from the storage
                blob=bucket.get_blob(f'Images/{id}.png')
                array=np.frombuffer(blob.download_as_string(),np.uint8)
                imgStudent=cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
                #update the data from attendance
                datetimeobject = datetime.strptime(studentinfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeobject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed > 30:
                    ref=db.reference(f'Students/{id}')
                    # Convert the integer value to a string before concatenating
                    studentinfo['total_attendance'] +=1
                    ref.child('total_attendance').set(studentinfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType=1
                    counter=0
                    imgMode = cv2.resize(imgModeList[modeType], (610, 988))
                    imgBackground[38:38 + 988, 858:858 + 610] = imgMode

            if modeType !=3:

                if 10<counter<20:
                    modeType=3
                    imgMode = cv2.resize(imgModeList[modeType], (610, 988))
                    imgBackground[38:38 + 988, 858:858 + 610] = imgMode


                if counter<=10:
                    cv2.putText(imgBackground,str(studentinfo['total_attendance']),(991,145),cv2.FONT_HERSHEY_COMPLEX,2,(50,50,50),3)
                    cv2.putText(imgBackground,str(studentinfo['name']),(1020,650),cv2.FONT_HERSHEY_COMPLEX,1,(10,10,10),2)
                    cv2.putText(imgBackground,str(id),(1011,775),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,50),2)

                    # center the resized image in the background
                    imgStudent_resized = cv2.resize(imgStudent, (310, 310))
                    imgBackground[220:220 + imgStudent_resized.shape[0], 1020:1020 + imgStudent_resized.shape[1]] = imgStudent_resized

                counter+=1

            if counter>=20:
                counter=0
                modeType=0
                studentinfo=[]
                imgStudent=[]
                imgMode = cv2.resize(imgModeList[modeType], (610, 988))
                imgBackground[38:38 + 988, 858:858 + 610] = imgMode
    else:
        modeType=0
        counter=0

    cv2.imshow("Face Attendance" , imgBackground)
    cv2.waitKey(1)