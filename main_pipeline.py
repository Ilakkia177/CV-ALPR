from car_detection import run_pipeline

# give all imp paths and start the pipeline for alpr.
if __name__ == "__main__":
    video_path = "data/cropped_10.mp4"
    templates_path = "templates/char_templates_hybrid"
    output_csv = "detected_plates_all.csv"

    run_pipeline(video_path, templates_path, output_csv)
