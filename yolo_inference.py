from ultralytics import YOLO
from ultralytics import settings


settings.update({"runs_dir": "runs"})

model = YOLO("models/yolov8x.pt")
# model = YOLO("models/yolo5_last.pt")

# result = model.predict("input_videos/image.png", save=True)
# result = model.predict("input_videos/input_video.mp4", conf=0.2, save=True)
result = model.track("input_videos/input_video.mp4", conf=0.2, save=True)


print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)
