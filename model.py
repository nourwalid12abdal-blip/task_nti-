

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

input_shape = (32, 32, 3) # Example input shape
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



x_train_preprocessed=0
y_train_categorical=0
model.fit(x_train_preprocessed, y_train_categorical, epochs=5, batch_size=128)
