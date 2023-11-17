import os
import modal
from dotenv import load_dotenv, dotenv_values

# REPLACE .env WITH YOUR OWN KEY_VALUE
config = dotenv_values(".env")
key_value = config["KEY"]
#print(key_value)

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       print(os.environ["HOPSWORKS_API_KEY"])
       g()


def generate_wine(typenumber, fixed_acidity_max, fixed_acidity_min, volatile_acidity_max, volatile_acidity_min, 
                    citric_acid_max, citric_acid_min, residual_sugar_max, residual_sugar_min, chlorides_max, chlorides_min, free_sulfur_dioxide_max, free_sulfur_dioxide_min, total_sulfur_dioxide_max, total_sulfur_dioxide_min, density_max, density_min, ph_max, ph_min, sulphates_max, sulphates_min, alcohol_max, alcohol_min, quality_min, quality_max):
    """
    Returns a wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "type": [typenumber],
                       "fixed_acidity": [random.uniform(fixed_acidity_max, fixed_acidity_min)],
                       "volatile_acidity": [random.uniform(volatile_acidity_max, volatile_acidity_min)],
                       "citric_acid": [random.uniform(citric_acid_max, citric_acid_min)],
                       "residual_sugar": [random.uniform(residual_sugar_max, residual_sugar_min)],
                       "chlorides": [random.uniform(chlorides_max, chlorides_min)],
                       "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide_max, free_sulfur_dioxide_min)],
                       "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_max, total_sulfur_dioxide_min)],
                       "density": [random.uniform(density_max, density_min)],
                       "ph": [random.uniform(ph_max, ph_min)],
                       "sulphates": [random.uniform(sulphates_max, sulphates_min)],
                       "alcohol": [random.uniform(alcohol_max, alcohol_min)],
                       "quality": [random.randint(quality_min, quality_max)],
                      })
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    redwine_df = generate_wine(0, 15.9, 4.6, 1.58, 0.12, 1.0, 0.0, 15.5, 0.9, 0.611, 0.012, 72.0, 1.0, 289.0, 6.0, 1.00369, 0.99007, 4.01, 2.74, 2.0, 0.33, 14.9, 8.4, 3, 8)
    whitewine_df = generate_wine(1, 14.2, 3.8, 1.1, 0.08, 1.66, 0.0, 65.8, 0.6, 0.346, 0.009, 289.0, 2.0, 440.0, 9.0, 1.03898, 0.98711, 3.82, 2.72, 1.08, 0.22, 14.2, 8.0, 3, 9)

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        wine_df = whitewine_df
        print("whitewine added")
    else:
        wine_df = redwine_df
        print("redwine added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(api_key_value=key_value)
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        with stub.run():
            print(f.remote())
