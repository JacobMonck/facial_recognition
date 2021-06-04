import numpy as np

import cv2

profile_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_profileface.xml")
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt.xml")


video = cv2.VideoCapture(0)

while True:
    read, frame = video.read()

    # Converts video to grey
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applies face cascade to frame
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x, y, w, h)

        color = (255, 0, 0)  # BGR NOT RGB
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h

        # Puts rectangle around faces
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Puts coordinate text around rectangle
        cv2.putText(
            frame,
            f"{x, y, w, h}",
            (end_cord_x, end_cord_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1
        )

    # Display the resulting frame
    grey_scale_bgr = np.dstack((grey, grey, grey))
    final_video = np.concatenate((frame, grey_scale_bgr), axis=1)
    cv2.imshow("Facial Recognition", final_video)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
