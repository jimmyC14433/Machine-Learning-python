import tensorflow as tf
import numpy as np

fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

'''Se define una red de dos capas con 3 unidades cada cada una, una entreda, una salida 
y una relacion densa entre cada nivel
'''
capa_oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
capa_oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa_oculta1, capa_oculta2, salida])

''' Calibrando el modelo de aprendizaje donde se definen
la metodologia de aprendizaje y las unidades de calibracion'''
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss='mean_squared_error')

print("Comenzando entrenamiento...")
historial = modelo.fit(fahrenheit, celsius, epochs=1000, verbose=False)
print("Modelo entrenado!")

print("Hagamos una predicción!")
g_celsius = int(input("Ingrese los grados fahrenheit: "))
resultado = round(int(modelo.predict([g_celsius])))
print(str(g_celsius) + "° fahrenheit equivale a " + str(resultado) + "° celsius!")

print("Variables internas del modelo")
print("Capa 1")
print(capa_oculta1.get_weights())
print("Capa 2")
print(capa_oculta2.get_weights())
print("Capa de salida")
print(salida.get_weights())