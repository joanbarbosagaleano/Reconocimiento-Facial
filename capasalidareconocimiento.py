import cv2
import os


dataruta='Documentos/Reconocimientofacial'
listdata=os.listdir(dataruta)
entrenamientomod1=cv2.face.EigenFaceRecognizer_create() 
entrenamientomod1.read('Entrenamientoneuronaeigen.xml')
ruido=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camara=cv2.VideoCapture(1)

while True:
    respuesta,captura=camara.read()
    if respuesta==False:
        print("ERROR CONECCIÓN CAMARA")
        break
    gris=cv2.cvtColor(captura,cv2.COLOR_BGR2GRAY)
    idcap=gris.copy()
    cara=ruido.detectMultiScale(gris,1.3,5)
    
    for(x,y,e1,e2) in cara:
        
        #cv2.rectangle(captura,(x,y),(x+e1,y+e2),(2,255,2),2)
        capturarostro=idcap[y:y+e2,x:x+e1]
        capturarostro=cv2.resize(capturarostro,(160,160),interpolation=cv2.INTER_CUBIC)
        resultado=entrenamientomod1.predict(capturarostro)
        cv2.putText(captura,'{}'.format(resultado),(x,y-5),1,1.3,(0,255,0),1,cv2.LINE_AA)
        
        if resultado[1]<8000:
            cv2.putText(captura,'{}'.format(listdata[resultado[0]]),(x,y-25),2,1.3,(255,255,0),1,cv2.LINE_AA)    
            cv2.rectangle(captura,(x,y),(x+e1,y+e2),(225,0,0),2)
        else:
            cv2.putText(captura,"NO ENCONTRADO",(x,y-20),2,1.3,(0,0,255),1,cv2.LINE_AA)    
            cv2.rectangle(captura,(x,y),(x+e1,y+e2),(225,0,0),2)

    cv2.imshow('Resultados',captura)    
    if cv2.waitKey(1)==ord('s'):
        break
camara.release()
cv2.destroyAllWindows()    