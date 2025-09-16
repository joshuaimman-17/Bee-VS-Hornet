import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Base directory
base_dir = 'kaggle_bee_vs_wasp'

# Load and filter the single labels.csv
csv_path = os.path.join(base_dir, 'labels.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file {csv_path} not found. Check unzip step.")

df = pd.read_csv(csv_path)
# Filter high-quality bee/wasp images
df = df[(df['photo_quality'] == 1) & (df['label'].isin(['bee', 'wasp']))]
# Map to binary labels (0: bee, 1: wasp/hornet)
df['label'] = df['label'].map({'bee': 0, 'wasp': 1})
# Prepend base_dir to relative paths (e.g., 'bee1/image.jpg' -> 'kaggle_bee_vs_wasp/bee1/image.jpg')
df['path'] = df['path'].apply(lambda x: os.path.join(base_dir, x.replace('\\', '/')))


# Split using flags
train_df = df[(df['is_validation'] == 0) & (df['is_final_validation'] == 0)]
val_df = df[df['is_validation'] == 1]
test_df = df[df['is_final_validation'] == 1]

# Print counts (expect train ~5k, val ~1.5k, test ~1.5k after filtering)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='path', y_col='label',
    target_size=(224, 224), batch_size=32, class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='path', y_col='label',
    target_size=(224, 224), batch_size=32, class_mode='raw'
)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary: bee (0) vs. hornet (1)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Unfreeze some layers for fine-tuning (optional for better accuracy)
for layer in base_model.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_generator, epochs=5, validation_data=val_generator)

# Evaluate on test
test_generator = val_datagen.flow_from_dataframe(
    test_df, x_col='path', y_col='label',
    target_size=(224, 224), batch_size=32, class_mode='raw', shuffle=False
)
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc}")

# Save model
model.save('bee_hornet_model.h5')

# Convert to TFLite for ESP32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('bee_hornet_model.tflite', 'wb') as f:
    f.write(tflite_model)