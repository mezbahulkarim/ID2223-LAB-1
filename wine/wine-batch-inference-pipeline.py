#REPLACE WITH YOUR OWN KEY VALUE
key_value = "ENTER VALUE"

# CHANGE THE IMAGE UPLOAD FUNCTIONALITY IF WANTED LATER 
#"modal token new"  ->      RUN THIS COMMAND IN VIRTUAL_ENV/CONDA  ONE TIME BEFORE RUNNING THIS FILE, it will link your modal account to this code
import os                       #overall this code will make inference, store predicted vs actual in feature store and show confusion matrix, also saves images to hopsworks resources
import modal

LOCAL=False      #LOCAL=False for running on modal etc.

if LOCAL == False:
   stub = modal.Stub("testing-wine-batch")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login(api_key_value=key_value)
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
                                                                                #FLOWER variable is prediction LABEL variable is actual/true value
    y_pred = model.predict(batch_data)
    print(y_pred)
    offset = 1
    pred_quality = y_pred[y_pred.size-offset]
    print("Quality predicted: " + str(pred_quality))
    
    #Not uploading any images, simply predicting wine quality no need
    #dataset_api = project.get_dataset_api()    
    #dataset_api.upload("./latest_iris.png", "Resources/images", overwrite=True)
   
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 
    print(df)
    true_quality = df.iloc[-offset]["quality"]
    print("Quality actual: " + str(true_quality))   
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Quality Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [pred_quality],
        'label': [true_quality],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)                                                 
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})           #INSERTS OUR PREDICTED vs ACTUAL 'data' value into the feature store
    
    history_df = monitor_fg.read()                                                  #SHOW recent history from wine-predictions feature group
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api = project.get_dataset_api()  
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]                                                  #history ends here

    # Create confusion matrix
    print("Number of different wine predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() >= 1:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results)
    
        pyplot.figure(figsize=(20, 10))
        cm = sns.heatmap(df_cm, annot=True, fmt="d")
        fig = cm.get_figure()
        fig.savefig("./wine_confusion_matrix.png")
        dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("Run the batch inference pipeline more than once at least") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            print(f.remote())           #CHANGED THIS TO SHOW PRINTS 

