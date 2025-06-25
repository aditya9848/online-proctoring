import cv2

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")
print("Model loaded ✅")
