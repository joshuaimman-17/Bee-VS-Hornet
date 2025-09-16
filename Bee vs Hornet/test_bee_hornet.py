import cv2
import numpy as np
import tensorflow.lite as tflite  # Changed from tflite_runtime
from playsound import playsound

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="bee_hornet_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default camera; change to 1 or other if needed
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Preprocess frame to match model input
    img = cv2.resize(frame, (224, 224))  # Match model input size
    input_data = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = output_data[0][0]  # Single value for binary classification
    label = 'Bee' if pred < 0.5 else 'Hornet'

    # Overlay text on frame
    color = (0, 255, 0) if label == 'Bee' else (0, 0, 255)  # Green for Bee, Red for Hornet
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Buzzer for hornet
    if label == 'Hornet':
        playsound('buzzer.wav')  # Ensure buzzer.wav is in the same folder

    # Display the frame
    cv2.imshow('Bee vs Hornet Detector (TFLite)', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()