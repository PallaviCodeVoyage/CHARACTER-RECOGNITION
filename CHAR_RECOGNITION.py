from mnist import MNIST
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
import cv2
from collections import deque
import time

# Load the data
data = MNIST(path='EMNIST_data/', return_type='numpy')
data.select_emnist('letters')
X, Y = data.load_training()

print(X.shape)
print(Y.shape)

# Reshape the data
X = X.reshape(-1, 28, 28)
Y = Y.reshape(-1, 1)

# Adjust the labels
Y = Y - 1

# Perform train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# Scale the pixel values to the range [0, 1]
X_test = X_test.astype('float32') / 255
X_train = X_train.astype('float32') / 255

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes=26)
Y_test = to_categorical(Y_test, num_classes=26)

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # Adjust dropout rate
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # Adjust dropout rate
model.add(Dense(26, activation='softmax'))  # Use 'softmax' for multi-class classification

# Compile the model with the Adam optimizer and 'categorical_crossentropy' loss
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Check accuracy before training
score = model.evaluate(X_test, Y_test, verbose=0)
accuracy = score[1] * 100
print("BEFORE TRAINING TEST ACCURACY ", accuracy)

# Train the model with checkpoints to save the best model
checkpoint_filename = 'best_model.keras'
checkpointer = ModelCheckpoint(filepath=checkpoint_filename, verbose=1, save_best_only=True)


# Fit the model with modified parameters
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)

# Loading model or weights
model.load_weights(checkpoint_filename)

# Calculate test accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
accuracy = 100 * score[1]
print("accuracy of the model is:", accuracy)

# Alphabet Recognition System
model = load_model(checkpoint_filename)

# Defining dictionary for letters
letters = {i: chr(97 + i) for i in range(26)}

# Defining blue color in HSV format
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])
kernel = np.ones((5, 5), np.uint8)

# Defining blackboard
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
alphabet = np.zeros((200, 200, 3), dtype=np.uint8)

# Deque to store alphabet drawn on screen
points = deque(maxlen=512)

# Opening the camera to recognize alphabet
cap = cv2.VideoCapture(0)
prediction = 0
last_update_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Flipping the image
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detecting which pixel values fall under blue color boundaries
    blue = cv2.inRange(hsv, blueLower, blueUpper)

    # Erosion
    blue = cv2.erode(blue, kernel)
    # Opening
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    # Dilation
    blue = cv2.dilate(blue, kernel)

    # Find contours of blue color object in the image
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    # If any contours were found
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        
        # Creating circle and finding center if contour is significant
        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), (125, 344, 278), 2)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            points.appendleft(center)
            last_update_time = time.time()
    else:
        # Only make predictions if no new points for 2 seconds
        if time.time() - last_update_time > 2 and len(points) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(blackboard_gray, 15)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            blackboard_cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

            if len(blackboard_cnts) >= 1:
                cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]

                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    alphabet = blackboard_gray[y - 10:y + h + 10, x - 10:x + w + 10]
                    try:
                        img = cv2.resize(alphabet, (28, 28))
                    except cv2.error as e:
                        continue

                    img = np.array(img)
                    img = img.astype('float32') / 255

                    prediction = model.predict(img.reshape(1, 28, 28))[0]
                    prediction = np.argmax(prediction)

            # Empty point deque and blackboard
            points = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

    # Connecting detected points with line
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
        cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

    cv2.putText(frame, "PREDICTION MADE :" + str(letters[int(prediction)]), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ALPHABET RECOGNITION SYSTEM", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
