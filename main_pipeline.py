from car_detection import run_pipeline

if __name__ == "__main__":
    video_path = "/Users/ilakkiat/Documents/MIT/SEM-V/CV-LAB/CV-PROJECT/CODE/data/cropped_10.mp4"
    templates_path = "/Users/ilakkiat/Documents/MIT/SEM-V/CV-LAB/CV-PROJECT/CODE/templates/char_templates_hybrid"
    output_csv = "/Users/ilakkiat/Documents/MIT/SEM-V/CV-LAB/CV-PROJECT/CODE/detected_plates_all.csv"

    run_pipeline(video_path, templates_path, output_csv)
