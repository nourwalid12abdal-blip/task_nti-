

<<<<<<< Updated upstream
base_model.trainable = False

""
=======
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_preprocessed, y_train_categorical, epochs=5, batch_size=128)
>>>>>>> Stashed changes
