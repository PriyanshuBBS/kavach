import cv2 as cv

# Since Haar is a sensitive


img = cv.imread('E:\data\Programming\Open CV\Photos\modi.jpeg') #read the image
cv.imshow('Modi',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier('E:\data\Programming\Face Recognition\haar.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

# Drawing rectangle over image
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)