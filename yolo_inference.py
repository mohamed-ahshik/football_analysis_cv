from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('/Users/user/Documents/personal_projects/football_analysis_cv/input_videos/08fd33_4.mp4', save=True)
# Get the bounding box for each object
# print(results[0])
# print('----------------------')
for box in results[0].boxes:
    # print(box)