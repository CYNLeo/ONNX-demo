from ultralytics import YOLO
import onnx


model = YOLO("infoBox.pt")

model.export(format="onnx",opset=17,half=True)

# model_onnx = onnx.load("infoBox.onnx")
# onnx.checker.check_model(model_onnx)

# print("Model is valid")