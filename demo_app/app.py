import pickle, gzip
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn



# File Paths
model_path = 'rf_model.sav'
endoing_path = "cat_encods.json"
component_config_path = "component_configs.json"
examples_path = "examples.pkl"

# predefined
target = "quality"

# predefined
feature_order = ['type', 'fixed acidity', 'volatile acidity', 'citric acid',
                 'residual sugar', 'chlorides', 'free sulfur dioxide', 
                 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# Loading the files
model = pickle.load(open(model_path, 'rb'))

# loading the examle cases for usage
with open("examples.pkl","rb") as f: examples = pickle.load(f)

# loading the classes & type casting the encoding indexes
classes = json.load(open(endoing_path, "r"))
classes = {k:{int(num):cat for num,cat in v.items() } for k,v in classes.items()}

inverse_class = {col:{val:key for key, val in clss.items()}  for col, clss in classes.items()}
labels = classes["quality"].values()

feature_limitations = json.load(open(component_config_path, "r"))

# Util function
def decode(col, data):
  return classes[col][data]

def encode(col, str_data):
  return inverse_class[col][str_data]

def feature_decode(df):

  # exclude the target var
  cat_cols = list(classes.keys())
  if "quality" in cat_cols:
    cat_cols.remove("quality")

  for col in cat_cols:
     df[col] = decode(col, df[col])

  return df

def feature_encode(df):
  
  # exclude the target var
  cat_cols = list(classes.keys())
  if "quality" in cat_cols:
    cat_cols.remove("quality")
  
  for col in cat_cols:
     df[col] = encode(col, df[col])
  
  return df

def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=feature_order)
  features = feature_encode(features)
  features = np.array(features).reshape(-1, len(feature_order))

  # prediction
  probabilities = model.predict_proba(features) #.predict(features)
  probs = probabilities.flatten()

  # output form
  results = {l : np.round(p, 3) for l, p in zip(labels, probs)}

  return results

# Creating the gui component according to component.json file
inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )



# creating the app
demo_app = gr.Interface(predict, inputs, "number",examples=examples)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()