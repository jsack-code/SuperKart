import streamlit as st
import pandas as pd
import requests

BACKEND_URL="http://backend:7860"

st.title("SuperKart Sales Prediction")

st.write("Input the product and store details to get a sales prediction")

# inputs for the user
Product_Weight = st.number_input("Product Weight")
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "High Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area")
Product_MRP = st.number_input("Product MRP")
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "Large"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "NC"])
Store_Age_Years = st.number_input("Store Age in Years")
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

# make JSON
outson={
  "Product_Weight": Product_Weight,
  "Product_Sugar_Content": Product_Sugar_Content,
  "Product_Allocated_Area": Product_Allocated_Area,
  "Product_MRP": Product_MRP,
  "Store_Size": Store_Size,
  "Store_Location_City_Type": Store_Location_City_Type,
  "Store_Type": Store_Type,
  "Product_Id_char": Product_Id_char,
  "Store_Age_Years": Store_Age_Years,
  "Product_Type_Category": Product_Type_Category
}

if st.button("Predict", type='primary'):
  response=requests.post(f"{BACKEND_URL}/v1_predict", json=outson)
else:
  st.error("No response from the API.")

st.subheader("Batch Prediction")
st.write(response.json())
upload=st.file_uploader("Upload a CSV file.",type=["csv"])
if upload is not None:
  if st.button("Batch Prediction",type='primary'):
    response=requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": upload})
    if response.status_code==200:
      results=response.json()
      st.success("Batch prediction successful!")
      try:
        if isinstance(results,list):
          df=pd.DataFrame(results)
        elif isinstance(results,dict):
          if all(not isinstance(v,(list,dict)) for v in results.values()):
            df=pd.DataFrame([results])
          else:
            df=pd.DataFrame(results)
        else: 
          df=pd.DataFrame({"Results": [results]})
        st.dataframe(df,use_container_width=True)
      except Exception as e:
        st.error(f"Error displaying results: {e}")
        st.json(results)
    else:
      st.error("Batch prediction failed, due to API error")
