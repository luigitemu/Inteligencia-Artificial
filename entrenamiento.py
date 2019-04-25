import numpy as np
import os
import re
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import load_img, img_to_array

# carga las imagenes 
dirname = os.path.join(os.getcwd(), 'entrenamiento')
imgpath = dirname + os.sep 

imagenes = []
directorios = []
dircount = []
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            imagenes.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directorios.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directorios))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

## crea las etiquetas para identificar todas las imagenes 
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
#print("Cantidad etiquetas creadas: ",len(labels))

## enmuera cada clase(deporte) a evaluar 
deportes=[]
indice=0
for directorio in directorios:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    deportes.append(name[len(name)-1])
    indice=indice+1


##
y = np.array(labels)
X = np.array(imagenes, dtype=np.uint8) #convierto de lista a numpy


classes = np.unique(y)# [0,1,2,3,4,5,6,7,8,9]
nClasses = len(classes)
##print('Total number of outputs : ', nClasses)
##print('Output classes : ', classes)

# preparo variables de entrenamiento 

train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.1) ## test size representa el porcentaje de imagenes que iran a prueba 
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# lo pasaremos  a codificacion one hot para las salidas sean de este modo [1,0,0,0,0,0,0,0,0]
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

#Mezclar todo y crear los grupos de entrenamiento y testing
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

#declaramos variables con los parámetros de configuración de la red
INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
epocas = 100 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria

#crear capas de la red neuronal , se creara una red neuronal convolucional 
# la cual es unna variacion del perceptron de multicapa 

sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2),padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(32, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5))
sport_model.add(Dense(nClasses, activation='softmax'))
#sport_model.summary()

#compilamos la red neuronal 
sport_model.compile(loss=keras.losses.categorical_crossentropy, 
                    optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100000), #cambio  lr 
                    metrics=['accuracy'])

#entrenamiento 
sport_train = sport_model.fit(  train_X, 
                                train_label, 
                                batch_size=batch_size,
                                epochs=epocas,
                                verbose=1,
                                validation_data=(valid_X,
                                 valid_label))
#guardamos el modelo 
sport_model.save("sports_mnist.h5py")
sport_model.save_weights('sport_weights.h5py')

