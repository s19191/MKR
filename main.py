import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
x = []
y = []
image_size = 150

for i in labels:
    folderPath = os.path.join('data', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        x.append(img)
        y.append(i)

x = np.array(x)
y = np.array(y)

x_train, y_train = shuffle(x, y, random_state=101)

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=101)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

efficientNet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

model = efficientNet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4, activation='softmax')(model)
model = tf.keras.models.Model(inputs=efficientNet.input, outputs=model)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

history = model.fit(x_train, y_train, validation_split=0.1, epochs=12, verbose=1, batch_size=32, callbacks=[tensorboard, checkpoint, reduce_lr])

pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

print(classification_report(y_test_new, pred))
