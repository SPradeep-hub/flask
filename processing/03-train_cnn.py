import os

# ✅ Force CPU BEFORE importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print('TensorFlow version:', tf.__version__)

# Check GPU
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# Paths
dataset_path = '.\\split_dataset\\'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

tmp_debug_path = '.\\tmp_debug'
os.makedirs(tmp_debug_path, exist_ok=True)

# Parameters
input_size = 128
batch_size_num = 32

# ✅ Data Generators (with correct preprocessing)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(input_size, input_size),
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(input_size, input_size),
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    classes=['real', 'fake'],
    target_size=(input_size, input_size),
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# ✅ Load EfficientNet (transfer learning)
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)

# ✅ Freeze base model (VERY IMPORTANT for small dataset)
for layer in efficient_net.layers:
    layer.trainable = False

# ✅ Build model
model = Sequential([
    efficient_net,
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# ✅ Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks
checkpoint_path = '.\\tmp_checkpoint'
os.makedirs(checkpoint_path, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ✅ Train
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

print(history.history)

# ✅ Load best model
best_model = load_model(os.path.join(checkpoint_path, 'best_model.keras'))

# ✅ Predict
test_generator.reset()

preds = best_model.predict(test_generator, verbose=1)

# ✅ Save results
results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})

print(results)