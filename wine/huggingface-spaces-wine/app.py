import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

key_value = "ENTER_VALUE"
project = hopsworks.login(api_key_value=key_value)
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[type,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,ph,sulphates,alcohol]], 
                      columns=['type','fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','ph','sulphates','alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)     
    result = res[res.size-1]
    return result
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with wine features to predict how good the wine quality is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1, label="wine type(Red wine= 0, White wine= 1 )"),
        gr.inputs.Number(default=7.2, label="fixed_acidity"),
        gr.inputs.Number(default=1.2, label="volatile_acidity"),
        gr.inputs.Number(default=1.2, label="citric_acid"),
        gr.inputs.Number(default=10.2, label="residual_sugar"),
        gr.inputs.Number(default=0.422, label="chlorides"),
        gr.inputs.Number(default=122.0, label="free_sulfur_dioxide"),
        gr.inputs.Number(default=122.0, label="total_sulfur_dioxide"),
        gr.inputs.Number(default=1.02323, label="density"),
        gr.inputs.Number(default=3.02, label="ph"),
        gr.inputs.Number(default=1.2, label="sulphates"),
        gr.inputs.Number(default=10.2, label="alcohol"),
        ],
    outputs="number",  # Set the output type to "number" for integer output
)

demo.launch(debug=True)

