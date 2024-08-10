import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Теперь TensorFlow доступен для использования в нашем коде на Python для создания и обучения нейронных сетей.
# Создание и обучение простой нейронной сети

# Давайте создадим и обучим простую полно-связную нейронную сеть на TensorFlow для решения задачи классификации.

# Сначала определяем архитектуру - последовательность слоев. Создадим 3 полно-связных слоя с 64, 32 и 10 нейронами:

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu')) 
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10)) 


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Далее обучим сеть на тренировочных данных:

# model.fit(train_images, train_labels, epochs=5)

# tensorboard = TensorBoard(log_dir="logs")

# model.fit(data, labels, epochs=10, callbacks=[tensorboard]) 


layer = model.layers[2]
# activations = layer.activations 

import matplotlib.pyplot as plt
# plt.imshow(activations[0][0,:,:], cmap='viridis')
#  новое добавление в код