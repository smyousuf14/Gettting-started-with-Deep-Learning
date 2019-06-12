from google.colab import drive
drive.mount("/content/gdrive")

from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("gdrive/My Drive/Colab Notebooks/idenprof-jpg/idenprof/models/model_ex-200_acc-0.849500.h5")
prediction.setJsonPath("gdrive/My Drive/Colab Notebooks/idenprof-jpg/idenprof/json/model_class.json")
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.predictImage("gdrive/My Drive/Colab Notebooks/firefighter.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
  print(eachPrediction , " : " , eachProbability)
