import numpy as np
import cv2
import wandb
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator


# Define Callback Function
class EpochDataCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Access the training metrics
        train_loss = logs['loss']
        train_acc = logs['accuracy']

        # Access the validation metrics
        val_loss = logs['val_loss']
        val_acc = logs['val_accuracy']

        wandb.log({
            'Epoch': epoch+1,
            'Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Test Loss': val_loss,
            'Test Accuracy': val_acc
        })

        print(f"\nEpoch {epoch+1} - Training Loss: {train_loss:.4f} - Training Accuracy: {train_acc:.4f} - "
              f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}\n")


# Build Model
def get_model():
    if model_name == '3-layer':
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
    elif model_name == 'vgg16':
        model = tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(48,48,1),
            pooling=None,
            classes=7,
            classifier_activation="softmax")
    elif model_name == 'vgg19':
        model = tf.keras.applications.VGG19(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(48,48,1),
            pooling=None,
            classes=7,
            classifier_activation="softmax")
    else:
        return NotImplementedError
    
    return model


# Train and Test Data Generator
def get_data_generator(batch_size=256, data_augmentation=False):
    train_dir = 'data/train'
    test_dir = 'data/test'

    if data_augmentation:
        train_data_generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.2,
            zoom_range=0.2)
    else:
        train_data_generator = ImageDataGenerator(rescale=1./255)
    
    test_data_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generator.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    test_generator = test_data_generator.flow_from_directory(
        test_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    return train_generator, test_generator


# Main Runnable Code for Model Training
model_name = 'vgg19'    # '3-layer', 'vgg19', 'vgg16'
data_augmentation = True
batch_size = 128
epoch_data_callback = EpochDataCallback()
save_name = f'{model_name}_with_augmentation' if data_augmentation else f'{model_name}_without_augmentation'
label_dict = {
    0: "Angry", 
    1: "Disgusted", 
    2: "Fearful", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprised"
}

# Wandb Init
wandb.init(project="Pattern-Recognition-Project", entity="kaykobad", name=save_name)

# Get Model and Data generators
model = get_model()
train_gen, test_gen = get_data_generator(batch_size=batch_size, data_augmentation=data_augmentation)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.0001, decay=1e-6),
    metrics=['accuracy'])
print(model.summary())

model_info = model.fit_generator(
        train_gen,
        steps_per_epoch=28709 // batch_size,
        epochs=100,
        validation_data=test_gen,
        validation_steps=7178 // batch_size,
        callbacks=[epoch_data_callback])

model.save_weights(f'{save_name}.h5')


