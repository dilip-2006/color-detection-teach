import cv2
import numpy as np
import json
import os

data_file = "taught_colors.json"
taught_colors = {}


if os.path.exists(data_file):
    with open(data_file, "r") as f:
        taught_colors = json.load(f)

def teach_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_pixel = hsv[y, x]
        print(f"\n Clicked HSV: {hsv_pixel}")

        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        lower = [max(h - 10, 0), max(s - 40, 0), max(v - 40, 0)]
        upper = [min(h + 10, 179), min(s + 40, 255), min(v + 40, 255)]

        color_name = input(" Enter name for this color (e.g., 'Red Cube'): ")
        taught_colors[color_name] = {"lower": lower, "upper": upper}

        with open(data_file, "w") as f:
            json.dump(taught_colors, f, indent=4)

        print(f" Saved color '{color_name}'!")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Teach Color")
cv2.setMouseCallback("Teach Color", teach_color)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Teach Color", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
