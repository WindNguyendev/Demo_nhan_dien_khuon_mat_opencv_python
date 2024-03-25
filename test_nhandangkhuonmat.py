# # Kho ảnh nhận dạng
# # 1 Load kho ảnh mẫu nhận dạng, chuyển BGR sang RGB
# # 2 Xác định vị trí khuôn mặt dùng thư viện (face_recognition.face_locations())
# # 3 Mã hóa các điểm trên khuôn mặt dùng thư viện ( face_recognition.face_encodings() )

# # Kho ảnh cần kiểm tra:
# # 1 Load kho ảnh cần kiểm tra, chuyển BGR sang RGB
# # 2 Xác định vị trí khuôn mặt 
# # 3 Mã hóa các điểm trên khuôn mặt dùng thư viện 


# # Check nhận dạng:
# # 1 so sánh encode_ảnh gốc, encode ảnh cần kiểm tra 
# #     face_recognition.compare_faces()
# # 2 Xác định khoảng cách( encode_ảnh gốc, encode_ảnh cần kiểm tra )
# #     face_recognition.face_distance





# import cv2
# import face_recognition

# imgVanh = face_recognition.load_image_file("pic/phong_1.jfif")
# imgVanh = cv2.cvtColor(imgVanh, cv2.COLOR_BGR2RGB)


# imgCheck = face_recognition.load_image_file("pic/phong_2.jfif")
# imgCheck = cv2.cvtColor(imgCheck, cv2.COLOR_BGR2RGB)


# #Xác định vị trí của khuôn mặt
# faceLoc = face_recognition.face_locations(imgVanh)[0]
# print(faceLoc) #(y1,x2,y2,x1)

# # Mã hóa hình ảnh
# encodeVanh = face_recognition.face_encodings(imgVanh)[0]
# cv2.rectangle(imgVanh,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


# faceCheck = face_recognition.face_locations(imgCheck)[0]
# print(faceCheck) #(y1,x2,y2,x1)


# encodeCheck = face_recognition.face_encodings(imgCheck)[0]
# cv2.rectangle(imgCheck,(faceCheck[3],faceCheck[0]),(faceCheck[1],faceCheck[2]),(255,0,255),2)


# results = face_recognition.compare_faces([encodeVanh],encodeCheck)
# print(results)


# # tuy nhiên nếu nhiều bức ảnh thì ta phải biết sai số là bao nhiêu
# faceDis = face_recognition.face_distance([encodeVanh],encodeCheck)

# print(results,faceDis)


# cv2.putText(imgCheck,f"{results}{1-(round(faceDis[0],2))}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
# cv2.imshow("Vân Anh",imgVanh)
# cv2.imshow("Check",imgCheck)

# cv2.waitKey()

import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path = "pic"

img = []
classname = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    img.append(curImg)
    classname.append(os.path.splitext(cl)[0])


print(classname)

def Mahoa(img):
    encodeList = []
    for i in img:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(i)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnow = Mahoa(img)
# print(len(encodeListKnow))


def thamdu(name):
    with open("thamdu.txt","r+") as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")




#Khởi động webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    framS = cv2.resize(frame,(0,0),None,fx =0.5,fy = 0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    #Xác định vị trí của khuôn mặt trên cam và encode nó
    facecurFrame = face_recognition.face_locations(framS) #lấy từng khuôn mặt và vị trí hiện tại
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame): #ấy từng khuôn mặt và vị trí hiện tại nhưng theo cặp
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if faceDis[matchIndex] < 0.50:
            name = classname[matchIndex].upper()
            thamdu(name)
        else:
            name = "Unknow"
        #in tên lên frame
        y1 ,x2, y2, x1 = faceLoc
        y1 ,x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow("Test lập trình nhận dạng khuôn mặt", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release() #giải phóng cam
cv2.destroyAllWindows()   




