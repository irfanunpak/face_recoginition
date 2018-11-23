import face_recognition
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pyautogui import press, typewrite, hotkey


def draw_face_landmarks(image, face):
	open_cv_image = None
	for face_landmarks in face_landmarks_list:
	    pil_image = Image.fromarray(image)
	    d = ImageDraw.Draw(pil_image, 'RGBA')
	    # Make the eyebrows into a nightmare
	    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
	    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
	    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
	    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

	    # # Gloss the lips
	    # d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
	    # d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
	    d.line(face_landmarks['top_lip'], fill=(0, 255, 0, 255), width=8)
	    d.line(face_landmarks['bottom_lip'], fill=(0, 255, 0, 255), width=8)

	    # Sparkle the eyes
	    d.polygon(face_landmarks['left_eye'], fill=(0, 255, 0, 255))
	    d.polygon(face_landmarks['right_eye'], fill=(0, 255, 0, 255))

	    # Apply some eyeliner
	    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
	    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

	    # pil_image.show()
	    open_cv_image = np.array(pil_image) 
	    # Convert RGB to BGR 
	    # open_cv_image = open_cv_image[:, :, ::-1].copy() 
	    # print(face_landmarks['top_lip'])
	return open_cv_image,face_landmarks['left_eye'],face_landmarks['right_eye'],face_landmarks['top_lip'],face_landmarks['bottom_lip']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 100000)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    # Display the resulting frame
    # cv2.imshow('frame',gray)
    # image = face_recognition.load_image_file(gray)
    # try:
    face_landmarks_list = face_recognition.face_landmarks(gray)
    try:
	    open_cv_image, left_eye, right_eye, top_lip, bottom_lip = draw_face_landmarks(gray, face_landmarks_list)
		# except:
		# 	print ('exception occur..')

	   
	    print(left_eye[1][1])
	    if left_eye[1][1] >= 309:
	    	press('up')
	    	print('blink',left_eye[0][1])
	    # if open_cv_image == None:
	    # 	cv2.imshow('frame',gray)
	    # else:
	    	# print(face_landmarks)
	    cv2.imshow('frame',open_cv_image)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
    except:
    	press('up')

    	print('wow')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
