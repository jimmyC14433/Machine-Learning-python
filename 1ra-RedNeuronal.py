"""   
            ESCENARIO
 En este escenario se plantea una red neuronal basica la cual se encargara de aprender a convertir 
 de grados celsius a fahrenheit con dos niveles entrada y salida, una sola relacion y un solo sesgo 
 """

import tensorflow as tf
import numpy as np

print("                CONVERTIDOR CELSIUS A FAHRENHEIT                ")

# Arrays con la información para entrenar la red neuroDDnal
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Calibrando el modelo de aprendizaje
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss="mean_squared_error")

print("Entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

print("Hagamos una predicción!")
g_celsius = int(input("Ingrese los grados celsius: "))
resultado = int(modelo.predict([g_celsius]))
print(str(g_celsius) + "° celsius equivale a " + str(resultado) + "° fahrenheit!")
