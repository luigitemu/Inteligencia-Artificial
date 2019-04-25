from tkinter import * 
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


modelo = 'sports_mnist.h5py'
pesos_modelo = 'sport_weights.h5py'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)
# funcion para probar el algoritmo 
def Prueba(archivo):
  x = load_img(archivo, target_size=(21, 28))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  resultado = array[0]
  respuesta = np.argmax(resultado)
  if respuesta == 0:
    mostrar = "pred: americano"
  elif respuesta == 1:
    mostrar = "pred: Basket"
  elif respuesta == 2:
    mostrar = "pred: baseball"
  elif respuesta == 3:
    mostrar = "pred: boxeo"
  elif respuesta == 4:
    mostrar = "pred: ciclismo"
  elif respuesta == 5:
    mostrar = "pred: formula 1"
  elif respuesta == 6:
    mostrar = "pred: futbol"
  elif respuesta == 7:
    mostrar = "pred: natacion"
  elif respuesta == 8:
    mostrar = "pred: tenis" 
  else:
    mostrar = 'desconocido'
  return mostrar



class Root(Tk): 
    def __init__(self):
        super(Root , self).__init__()
        self.title('Identificador de Deportes')
        self.minsize(700 , 200)
        self.labelFrame = ttk.LabelFrame(self , text= 'Abre una Imagen')
        self.labelFrame.grid(column = 0 , row = 0 , padx= 20 , pady= 30)
        self.button()
        self.mainloop()
        

    def button(self): 
        self.button = ttk.Button(self.labelFrame , text = 'ingrese la imagen a analizar ', command = self.fileDialog)
        self.button.grid(column = 1 , row = 1 )

    def fileDialog(self): 
        self.label = ttk.Label(self.labelFrame , text = '')
        self.filename = filedialog.askopenfilename(initialdir = '/' , title = ' seleccione una imagen' , filetype = (('jpeg files' , '*.jpg') , ('All Files' , '*.*')) )
        
        self.label.grid(column=1 , row = 2)
        deporte = Prueba(self.filename)
        self.label.update()
        self.label.configure( text = deporte)
        
        #self.directorio = ' '
        #self.directorio = self.filename
        #print(self.directorio)

raiz = Root()