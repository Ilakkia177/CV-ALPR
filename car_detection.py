import cv2
import numpy as np
import csv
import os
from datetime import datetime
from plate_recognition import extract_plate_text
from stabilization import get_stabilized_plate
from template_loader import load_templates

def run_pipeline(video_path, templates_path, output_csv):
    resize_width = 800
    motion_threshold = 2.5
    min_area = 100000
    aspect_ratio_min, aspect_ratio_max = 1.0, 4.0
    min_width, min_height = 60, 40
    stabilization_window = 8
    min_confident_detections = 1

    cap = cv2.VideoCapture(video_path)
    os.makedirs("/Users/ilakkiat/Documents/MIT/SEM-V/CV-LAB/CV-PROJECT/CODE/output_cars", exist_ok=True)

    # with open(output_csv, "w", newline="") as f:
    #     csv.writer(f).writerow(["Frame", "Timestamp", "Plate"])
    with open(output_csv, "w", newline="") as f:
      csv.writer(f).writerow(
          ["Frame", "Car_X", "Car_Y", "Car_W", "Car_H",
          "Plate_X", "Plate_Y", "Plate_W", "Plate_H",
          "Plate"]
      )


    template_dict = load_templates(templates_path)
    plate_history, prev_stabilized = [], None

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Cannot read video")
        cap.release()
        return

    scale = resize_width / frame1.shape[1]
    frame1 = cv2.resize(frame1, (resize_width, int(frame1.shape[0] * scale)))
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_count = 0

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        frame_count += 1
        frame2 = cv2.resize(frame2, (resize_width, int(frame2.shape[0] * scale)))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = np.uint8((mag > motion_threshold) * 255)
        kernel = np.ones((7, 7), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_plate = None

        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if not (aspect_ratio_min <= ar <= aspect_ratio_max):
                continue
            if w < min_width or h < min_height:
                continue

            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            car_roi = frame2[y:y + h, x:x + w]
            cv2.imwrite(f"/Users/ilakkiat/Documents/MIT/SEM-V/CV-LAB/CV-PROJECT/CODE/output_cars/car_{frame_count}_{x}_{y}.jpg", car_roi)

            gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
            gray_car = cv2.bilateralFilter(gray_car, 11, 17, 17)
            edges = cv2.Canny(gray_car, 30, 200)
            cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts:
                
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if len(approx) == 4:
                    x2, y2, w2, h2 = cv2.boundingRect(approx)
                    ar_p = w2 / float(h2)
                    if 3.0 < ar_p < 5.0 and 2500 < cv2.contourArea(cnt) < 20000 and y2 > car_roi.shape[0] // 2:
                        cv2.rectangle(car_roi, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                        plate_roi = car_roi[y2:y2 + h2, x2:x2 + w2]
                        plate_number = extract_plate_text(plate_roi, template_dict)
                        if plate_number:
                            detected_plate = plate_number

        if detected_plate:
            plate_history.append(detected_plate)
            if len(plate_history) > stabilization_window:
                plate_history.pop(0)

        stabilized_plate = get_stabilized_plate(plate_history)

        if stabilized_plate and len(plate_history) >= min_confident_detections:
            if stabilized_plate != prev_stabilized:
                # timestamp = datetime.now().strftime("%H:%M:%S")
                with open(output_csv, "a", newline="") as f:
                  csv.writer(f).writerow([
                      frame_count, x, y, w, h, x2, y2, w2, h2, stabilized_plate
                  ])

                prev_stabilized = stabilized_plate
                print(f" Saved stabilized plate: {stabilized_plate}")

        if stabilized_plate:
            label = stabilized_plate
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x0, y0 = 30, 50
            cv2.rectangle(frame2, (x0 - 10, y0 - th - 10), (x0 + tw + 10, y0 + 10), (0, 0, 0), -1)
            cv2.putText(frame2, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Car + Plate Detection + Stabilized Multi-Plate Logging", frame2)
        prev_gray = gray.copy()

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nAll stabilized plates saved successfully.")