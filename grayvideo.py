import cv2
cap=cv2.VideoCapture(0)
while True:
	ret,frame=cap.read()
	grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret==False:
		continue
	cv2.imshow("videoframe",frame)
	cv2.imshow("grayframe",grayframe)
	key_pressed=cv2.waitKey(1) & 0XFF
	if key_pressed==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()