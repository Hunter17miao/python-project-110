# To Capture Frame
import cv2
# To process image array
import numpy as np
# import the tensorflow modules and load the model
import tensorflow as tf

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)
model = tf.keras.models.load_model('keras_model.h5')
# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera
    status, frame = camera.read()
    # if we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Resize the frame
        img = cv2.resize(frame, (224, 224))
        # Expand the dimensions
        test_image = np.array(img, dtype=np.float32)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Normalize it before feeding to the model
        normalized_image = test_image / 255.0
        # Get predictions from the model
        prediction = model.predict(normalized_image)
        print(prediction)
        
        # Displaying the frames captured
        cv2.imshow('feed', frame)
        
        # Waiting for 1ms
        key = cv2.waitKey(1)
        
        # If space key is pressed, break the loop
        if key == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()