import cv2
import os

# Create a directory to store frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

fps_list = [25, 50]  # List of frame rates to display
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame
    frame_filename = f'frames/frame_{frame_count:04d}.jpg'
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

    # Display the frame
    cv2.imshow('Frame', frame)

    # Wait for the specified time based on fps
    key = cv2.waitKey(int(1000 / fps_list[0])) & 0xFF  # Change fps_list[0] to fps_list[1] for 50 fps

    # Press 'q' to exit the loop
    if key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()