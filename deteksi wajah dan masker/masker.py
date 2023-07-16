import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('Nariz_Hidung.xml')

video_capture = cv2.VideoCapture(0)
mask_on = False

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#deteksi wajah

    face = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in face:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        if mask_on:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText (frame,'Mask on', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)
            # os.system("start alarm.M4A")
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText (frame,'Mask off', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 5)
            

        #Deteksi mata
        # eye = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eye:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        #     cv2.putText (frame,'eye', (x + ex,y +ey), 1, 2,(0, 255, 0), 2)
            

    #Deteksi hidung 
    nose = nose_cascade.detectMultiScale(gray, 1.18, 35,)
    for (sx,sy,sw,sh) in nose:
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0) , 2)
        cv2.putText (frame,'nose', (x + sx,y +sy), 1, 1,(0, 255, 0), 2)

    if len(nose)>0:
        mask_on = False
    else:
        mask_on = True

    cv2.putText (frame,'jumlah wajah : ' + str(len(face)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
