import cv2
import numpy as np

def preProcessImg(img,inputSize=(416,416)):
    h, w,_ = img.shape
    ih, iw = inputSize
    scale = min(iw / w, ih / h)
    nw = int(w * scale)
    nh = int(h * scale)
    imgR = cv2.resize(img,(nw, nh),cv2.INTER_CUBIC)
    newImg=cv2.copyMakeBorder(imgR,(ih-nh)//2,ih-nh-(ih-nh)//2,(iw-nw)//2,iw-nw-(iw-nw)//2,
                       cv2.BORDER_CONSTANT,(128, 128, 128))
    imgR=newImg/255.0
    dst=np.expand_dims(imgR,axis=0)
    return dst

if __name__=='__main__':
    img=cv2.imread('./demo/person.jpg')
    print(img.shape)
    nImg=preProcessImg(img)
    print(nImg.shape)
    cv2.imshow('test',nImg)
    cv2.waitKey()