import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('Nariz_Hidung.xml')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#deteksi wajah

    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText (frame,'face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
       
       #Deteksi mata
        eye = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText (frame,'eye', (x + ex,y +ey), 1, 2,(0, 255, 0), 2)
            
            #Deteksi hidung 
            nose = nose_cascade.detectMultiScale(gray, 1.18, 35)
            for (sx,sy,sw,sh) in nose:
                cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0) ,2)
                cv2.putText (frame,'nose', (x + sx,y +sy), 1, 3,(255, 0, 0), 2)

    cv2.putText (frame,'jumlah wajah : ' + str(len(face)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
    cv2.imshow('Face', frame)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
