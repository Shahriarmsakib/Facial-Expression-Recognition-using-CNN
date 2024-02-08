import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


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

# Main Runnable Code for Model Training
model_name = '3-layer'    # '3-layer', 'vgg19', 'vgg16'
data_augmentation = True

# Get Model and Data generators
model = get_model()
model.load_weights('weights/3-layer_with_augmentation.h5')

# List of Emotions
emotions = {
    0: "   Angry   ", 
    1: "Disgusted", 
    2: "  Fearful  ", 
    3: "   Happy   ", 
    4: "  Neutral  ", 
    5: "    Sad    ", 
    6: "Surprised"
}

emojis = {
    0: "./emojis/angry.png", 
    1: "./emojis/disgusted.png", 
    2: "./emojis/fearful.png", 
    3: "./emojis/happy.png", 
    4: "./emojis/neutral.png", 
    5: "./emojis/sad.png", 
    6:"./emojis/surpriced.png"
}

# Open Camera
cv2.ocl.setUseOpenCL(False)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera")
    exit()

# Global Variable
recent_frame = np.zeros((640, 480, 3), dtype=np.uint8)
detected_emotion = emotions[0]
selected_emoji = emojis[0]
emoji = cv2.imread(selected_emoji)

while True:
    _, frame = camera.read()
    frame = cv2.resize(frame, (640, 480))

    # Get Bounding Box
    monochrome_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b_box = cv2.CascadeClassifier('weights/haarcascade_frontalface_default.xml')
    num_faces = b_box.detectMultiScale(monochrome_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        # Mark the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # Region of interest fitting
        roi = monochrome_frame[y:y+h, x:x+w]
        # cv2.imshow("roi", cv2.resize(roi, (48, 48)))
        prediction = model.predict(np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0))

        # Prediction from model
        maxindex = int(np.argmax(prediction))
        detected_emotion = emotions[maxindex]
        selected_emoji = emojis[maxindex]
        emoji = cv2.imread(selected_emoji)
        print(f"Detected Emotion: {maxindex} - {detected_emotion}")

    # Show the frames with emoji
    cv2.imshow("Live Feed", frame)
    cv2.putText(emoji, detected_emotion, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Detected Emotion", emoji)

    # Pause for 0.3 second
    key = cv2.waitKey(300)

    # Stop and break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all resources
camera.release()
cv2.destroyAllWindows()
