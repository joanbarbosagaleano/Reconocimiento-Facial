#Capa facial
import cv2 as cv
import numpy as np
import os 

modelo='Elon Musk'
ruta1='Documentos/Reconocimientofacial'
rutacompleta=ruta1+'/'+ modelo
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)


camara=cv.VideoCapture('reconocimientofacial1/ElonMusk.mp4')
ruido=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#_,cap=camara.read()
#cv.imshow("camara",cap)

id=0
while True:
    respuesta,captura=camara.read()
    if respuesta==False:
        break
    gris=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcap=captura.copy()
    cara=ruido.detectMultiScale(gris,1.3,5)
    
    
    for(x,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(2,255,2),2)
        capturarostro=idcap[y:y+e2,x:x+e1]
        capturarostro=cv.resize(capturarostro,(160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id),capturarostro)
        id+=1

  #  cv.imshow("resultado rostro",captura)    

    if id==500:
        break
camara.release()
cv.destroyAllWindows()
    