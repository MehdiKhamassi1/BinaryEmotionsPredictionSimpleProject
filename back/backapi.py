from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost:4200",  
    "http://127.0.0.1:4200",   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
from tensorflow.keras.models import load_model
model = load_model('../imageclassifier.h5')

def prepare_image(image_bytes: bytes):
    
    image = Image.open(io.BytesIO(image_bytes))
    
    image = image.resize((256, 256))
    # Convert image to numpy array
    image_array = np.array(image) / 255.0 
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image data from the uploaded file
    #image_data = await file.read()
    #image_array = prepare_image(image_data)    
    # Predict the class of the image
    #predictions = model.predict(image_array)   
    # Assuming the model has a softmax output and the predictions are probabilities
    #predicted_class = np.argmax(predictions, axis=1)
    image_data = await file.read()
    image_array = prepare_image(image_data)
    yhat = model.predict(image_array)
    print(yhat)
    predicted_class = int(yhat[0][0] > 0.5)
    # Return the predicted class
    #return JSONResponse(content={"predicted_class": int(predicted_class[0])})
    return {"prediction": int(predicted_class)}

