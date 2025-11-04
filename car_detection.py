import cv2
import numpy as np
import csv
import os
from plate_recognition import extract_plate_text 
from stabilization import get_stabilized_plate
from template_loader import load_templates

def run_pipeline(video_path, templates_path, output_csv):
    resize_width = 800
    motion_threshold = 2.5
    min_area = 100000 #to filter out false positives, testwd
    aspect_ratio_min, aspect_ratio_max = 1.0, 4.0 # car bb aspect ratio limits
    min_width, min_height = 60, 40 # for car
    stabilization_window = 8 # number of past detection used for stabilisation
    min_confident_detections = 2. # how many stable detections needed before saving


    cap = cv2.VideoCapture(video_path)
    os.makedirs("output_cars", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
      csv.writer(f).writerow(
          ["Frame", "Car_X", "Car_Y", "Car_W", "Car_H",
          "Plate_X", "Plate_Y", "Plate_W", "Plate_H",
          "Plate"]
      )


    template_dict = load_templates(templates_path)

    # list to store recent plate detections
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

        # resize,grayscale
        frame2 = cv2.resize(frame2, (resize_width, int(frame2.shape[0] * scale)))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        #calculate optical flow between  frames
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])  #motion magni
        motion_mask = np.uint8((mag > motion_threshold) * 255)  # bi mask for moving areas
        

        # cleaning  motion mask using morphology
        kernel = np.ones((7, 7), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel)

        # find contour of moving car
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_plate = None
        plate_box = None

        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue

            #bb around contour-car
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)

            #filter criteria
            if not (aspect_ratio_min <= ar <= aspect_ratio_max):
                continue
            if w < min_width or h < min_height:
                continue

            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            car_roi = frame2[y:y + h, x:x + w]

            #save cars to analyze
            cv2.imwrite(f"CV-PROJECT/CODE/output_cars/car_{frame_count}_{x}_{y}.jpg", car_roi)

            #preprocess car to find the plate
            gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
            gray_car = cv2.bilateralFilter(gray_car, 11, 17, 17) #edge-preserve smoothining
            edges = cv2.Canny(gray_car, 30, 200) #detect edges
            cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts:


                # approximate contour to polygon and check if rectangular(cause rectangular plate)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if len(approx) == 4:

                    x2, y2, w2, h2 = cv2.boundingRect(approx)
                    ar_p = w2 / float(h2)

                    #filter plate characteristics
                    if 3.0 < ar_p < 5.0 and 2500 < cv2.contourArea(cnt) < 20000 and y2 > car_roi.shape[0] // 2:
                        plate_roi = car_roi[y2:y2 + h2, x2:x2 + w2]
                        plate_number = extract_plate_text(plate_roi, template_dict)
                        if plate_number:
                            detected_plate = plate_number
                            plate_box = (x2, y2, w2, h2)
                            cv2.rectangle(car_roi, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                        plate_roi = car_roi[y2:y2 + h2, x2:x2 + w2]

                        # recognition of characters
                        plate_number = extract_plate_text(plate_roi, template_dict)
                        if plate_number:
                            detected_plate = plate_number


        # add detected plate to history for stabilization
        if detected_plate:
            plate_history.append(detected_plate)
            if len(plate_history) > stabilization_window:
                plate_history.pop(0)

        #stabilize the plates recognition output
        stabilized_plate = get_stabilized_plate(plate_history)

        if stabilized_plate and len(plate_history) >= min_confident_detections:
            if stabilized_plate != prev_stabilized:

                if plate_box is not None:
                    x2, y2, w2, h2 = plate_box
                    abs_x2 = x + x2
                    abs_y2 = y + y2
                    abs_w2 = w2
                    abs_h2 = h2


                # Scale all coordinates back to original frame size
                orig_x  = int(x  / scale)
                orig_y  = int(y  / scale)
                orig_w  = int(w  / scale)
                orig_h  = int(h  / scale)
                orig_x2 = int(abs_x2 / scale)
                orig_y2 = int(abs_y2 / scale)
                orig_w2 = int(abs_w2 / scale)
                orig_h2 = int(abs_h2 / scale)

                # Write to CSV
                with open(output_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        frame_count, orig_x, orig_y, orig_w, orig_h,
                        orig_x2, orig_y2, orig_w2, orig_h2, stabilized_plate
                    ])
                
                
                  

                prev_stabilized = stabilized_plate
                print(f" Saved stabilized plate: {stabilized_plate}")


        # display stabilized plate on video
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