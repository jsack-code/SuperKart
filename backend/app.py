import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api=Flask("Superkart")
model=joblib.load("backend_files/xgboost_tuned.joblib")

# make a homepage
@superkart_api.get('/')
def home():
  return "Welcome to SuperKart"

# make a single product prediction space
@superkart_api.post('/v1/predict')
def predict_one():
  data=request.get_json()
  sample={
      "Product_Weight": data["Product_Weight"],
      "Product_Sugar_Content": data["Product_Sugar_Content"],
      "Product_Allocated_Area": data["Product_Allocated_Area"],
      "Product_MRP": data["Product_MRP"],
      "Store_Size": data["Store_Size"],
      "Store_Location_City_Type": data["Store_Location_City_Type"],
      "Store_Type": data["Store_Type"],
      "Product_Id_char": data["Product_Id_char"],
      "Store_Age_Years": data["Store_Age_Years"],
      "Product_Type_Category": data["Product_Type_Category"]
  }

  input_data=pd.DataFrame([sample])
  prediction=model.predict(input_data)
  return jsonify({'prediction': list(prediction)})

# make a space for batch prediction
@superkart_api.post('/v1/predictbatch')
def predict_batch():
  file=request.files['file']
  input_data=pd.read_csv(file,index_col=0)
  prediction=model.predict(input_data).tolist()
  return jsonify({'prediction': prediction})
  decoder={str(i): round(pred,2) for i, pred in enumerate(prediction)}
  return decoder

if __name__=="__main__":
  superkart_api.run(debug=True)
