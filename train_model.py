# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# import os

# # --- Configuration ---
# # MobileNetV2 was trained on 224x224, but 160x160 is a good balance
# IMG_WIDTH, IMG_HEIGHT = 160, 160 
# BATCH_SIZE = 32
# EPOCHS = 10 # Start with 10, you can increase to 15 or 20 for better accuracy
# # This path must match your folder structure
# TRAIN_DIR = 'dataset/train' 

# # --- 1. Prepare Data Augmentation ---
# # Create a data generator that rescales images and applies augmentations.
# # This creates more robust training data by modifying images on the fly.
# # We also split the data: 80% for training, 20% for validation.
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.2 # Use 20% of data for validation
# )

# # --- 2. Load Data from Directories ---
# print("Loading training data...")
# train_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_WIDTH, IMG_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='training' # Set as training data
# )

# print("Loading validation data...")
# validation_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_WIDTH, IMG_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation' # Set as validation data
# )

# # --- 3. Build the Model using MobileNetV2 (Transfer Learning) ---
# # Load the MobileNetV2 model, pre-trained on ImageNet, without its top classification layer.
# base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
#                          include_top=False, # Don't include the final ImageNet classifier
#                          weights='imagenet')

# # Freeze the base model layers so we don't retrain them.
# # We only want to train our new custom layers.
# base_model.trainable = False

# # Create our new custom layers on top of the base model.
# x = base_model.output
# x = GlobalAveragePooling2D()(x) # A layer to average the features.
# x = Dense(128, activation='relu')(x) # A fully-connected layer for learning.
# x = Dropout(0.5)(x) # Dropout helps prevent overfitting.
# # The final output layer with one neuron (for cat/dog) and sigmoid activation.
# predictions = Dense(1, activation='sigmoid')(x) 

# # This is our final model.
# model = Model(inputs=base_model.input, outputs=predictions)

# # --- 4. Compile the Model ---
# # We use a low learning rate to ensure the model learns effectively.
# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# print("Model Summary:")
# model.summary()


# # --- 5. Train the Model ---
# print("\nStarting training...")
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     validation_data=validation_generator
# )

# # --- 6. Save the Final Model ---
# # Save it with a new name to avoid overwriting your old one.
# model.save('cat_dog_model_v2.keras')
# print("\nTraining complete! New model saved as 'cat_dog_model_v2.keras'")


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# import os

# # --- Configuration ---
# IMG_WIDTH, IMG_HEIGHT = 160, 160 
# BATCH_SIZE = 32
# # We will do an initial training phase, then a fine-tuning phase
# INITIAL_EPOCHS = 20
# FINE_TUNE_EPOCHS = 10
# TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# # This path must match your folder structure
# TRAIN_DIR = 'dataset/train' 

# # --- 1. Prepare Data Augmentation ---
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.2 # Use 20% of data for validation
# )

# # --- 2. Load Data from Directories ---
# print("Loading training data...")
# train_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_WIDTH, IMG_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='training'
# )

# print("Loading validation data...")
# validation_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_WIDTH, IMG_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation'
# )

# # --- 3. Build the Model using MobileNetV2 ---
# base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
#                          include_top=False,
#                          weights='imagenet')

# # Freeze the base model layers initially
# base_model.trainable = False

# # Create our new custom layers on top
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(1, activation='sigmoid')(x) 

# model = Model(inputs=base_model.input, outputs=predictions)

# # --- 4. Compile the Model for Initial Training ---
# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# print("--- Initial Model Summary (base frozen) ---")
# model.summary()

# # --- 5. Initial Training ---
# print(f"\n--- Starting Initial Training for {INITIAL_EPOCHS} epochs ---")
# history = model.fit(
#     train_generator,
#     epochs=INITIAL_EPOCHS,
#     validation_data=validation_generator
# )

# # --- 6. Prepare for Fine-Tuning ---
# # Unfreeze the top layers of the model. We will train the last few blocks.
# base_model.trainable = True

# # Let's unfreeze from layer 100 onwards.
# fine_tune_at = 100 

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False

# # Re-compile the model with a very low learning rate for fine-tuning
# model.compile(optimizer=Adam(learning_rate=1e-5), # 10x lower learning rate
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# print("\n--- Fine-Tuning Model Summary (top layers unfrozen) ---")
# model.summary()

# # --- 7. Fine-Tune the Model ---
# print(f"\n--- Starting Fine-Tuning for {FINE_TUNE_EPOCHS} epochs ---")
# history_fine = model.fit(
#     train_generator,
#     epochs=TOTAL_EPOCHS,
#     initial_epoch=history.epoch[-1], # Continue from where we left off
#     validation_data=validation_generator
# )

# # --- 8. Save the Final, Fine-Tuned Model ---
# model.save('cat_dog_model_v3_finetuned.keras')
# print("\nFine-tuning complete! New model saved as 'cat_dog_model_v3_finetuned.keras'")


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
import json

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 160, 160 
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
TRAIN_DIR = 'dataset/train' 

# --- 1. Prepare Data Augmentation ---
print("Setting up data augmentation...")

# First, let's count the number of images in each class manually
def count_images_per_class(directory):
    classes = os.listdir(directory)
    class_counts = {}
    for class_name in classes:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            class_counts[class_name] = num_images
    return class_counts

class_counts = count_images_per_class(TRAIN_DIR)
print(f"Class counts: {class_counts}")

# Calculate class weights manually
total_images = sum(class_counts.values())
class_weights = {}
for i, (class_name, count) in enumerate(class_counts.items()):
    class_weights[i] = total_images / (len(class_counts) * count) if count > 0 else 1.0

print(f"Class weights: {class_weights}")

# Save class weights for later use
with open('class_weights.json', 'w') as f:
    json.dump(class_weights, f)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

# --- 2. Load Data ---
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

print("Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Classes: {train_generator.class_indices}")

# --- 3. Build Enhanced Model ---
print("Building model...")
base_model = MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet',
    alpha=1.0
)

base_model.trainable = False

# Enhanced top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile with Callbacks ---
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.2, patience=3, verbose=1),
    ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1)
]

print("--- Model Summary ---")
model.summary()

# --- 5. Initial Training ---
print(f"\n--- Starting Initial Training for {INITIAL_EPOCHS} epochs ---")
history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# --- 6. Fine-Tuning ---
print("\n--- Preparing for Fine-Tuning ---")
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Count trainable layers
trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"Number of trainable layers: {trainable_count}")

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("\n--- Starting Fine-Tuning ---")
history_fine = model.fit(
    train_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# --- 7. Save Final Model ---
model.save('cat_dog_model_improved.keras')
print("\nTraining complete! Model saved as 'cat_dog_model_improved.keras'")

# --- 8. Evaluate Model ---
print("\n--- Final Evaluation ---")
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(validation_generator, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.2%}")
print(f"Validation Precision: {val_precision:.2%}")
print(f"Validation Recall: {val_recall:.2%}")

# Calculate F1 score
if val_precision + val_recall > 0:
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    print(f"Validation F1 Score: {val_f1:.2%}")

print("\n--- Training Complete! ---")