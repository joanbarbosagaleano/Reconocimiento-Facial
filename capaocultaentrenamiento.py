import cv2 as cv
import os
import numpy as np
from time import time
dataruta='Documentos/Reconocimientofacial'
listdata=os.listdir(dataruta)

ids=[]
rostrosdata=[]
id=0
timeinicial=time()
for fila in listdata:
    rutacompleta=dataruta+'/'+fila
    print('Iniciando lectura...')
    for archivo in os.listdir(rutacompleta):
        
        print('imagenes: ',fila+'/'+archivo)
        ids.append(id)
        rostrosdata.append(cv.imread(rutacompleta+'/'+archivo,0)) #el cero al final es para pasar a escala grises
        
    id+=1
    tiempofinal=time()
    tiempototal=tiempofinal-timeinicial    
    print('Tiempo total lectura: ',tiempototal)

entrenamientomod1=cv.face.EigenFaceRecognizer_create() 
print('Iniciando el entrenamiento ...espere')
entrenamientomod1.train(rostrosdata,np.array(ids))
tiempofinalentrenamiento=time()
tiempototalentrenamiento=tiempofinalentrenamiento-tiempofinal
print('tiempo entrenamiento total: ',tiempototalentrenamiento)

entrenamientomod1.write('Entrenamientoneuronaeigen.xml')
print('Entrenamiento concluido')