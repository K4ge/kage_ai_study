from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")   # 小模型，适合先试玩

# 1) 跑图片
results = model("test.jpg", save=True)
print(results)

# 2) 开摄像头实时识别
# 把上面注释掉，改成下面这行
# results = model.predict(source=0, show=True)