import gradio as gr
from PIL import Image
import hopsworks
import pandas as pd
import hopsworks
import joblib
import datetime
from datetime import datetime
import dataframe_image as dfi
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import seaborn as sns
import requests
from dotenv import load_dotenv, dotenv_values
import random


key_value = "KEY_VALUE"

project = hopsworks.login(api_key_value=key_value)
fs = project.get_feature_store()
    
mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
    
feature_view = fs.get_feature_view(name="wine", version=1)
batch_data = feature_view.get_batch_data()
                                                                                
y_pred = model.predict(batch_data)
print(y_pred)
print("----------------------------------------------------------------------------------")
# Setting Offset Value    
offset = y_pred.shape[0]
print(f'Offset: {offset}')              # number of rows
offset = offset -1                      # account for index value
random_offset = random.randint(0, offset)

pred_quality = y_pred[random_offset]

wine_fg = fs.get_feature_group(name="wine", version=1)
df = wine_fg.read() 
print(df)
true_quality = df.iloc[-random_offset]["quality"]

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/images/wine_df_recent.png", overwrite=True)
dataset_api.download("Resources/images/wine_confusion_matrix.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted quality")
          input_value = gr.Text(pred_quality)
      with gr.Column():          
          gr.Label("Today's Actual quality")
          input_value = gr.Text(true_quality)        
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("wine_df_recent.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("wine_confusion_matrix.png", elem_id="confusion-matrix")        

demo.launch(share=True)
