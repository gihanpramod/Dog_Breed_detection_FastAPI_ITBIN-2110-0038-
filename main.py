from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("dogclassification.h5")

# Define class names
class_names = {
    "0": "Afghan", "1": "African Wild Dog", "2": "Airedale", "3": "American Hairless",
    "4": "American Spaniel", "5": "Basenji", "6": "Basset", "7": "Beagle",
    "8": "Bearded Collie", "9": "Bermaise", "10": "Bichon Frise", "11": "Blenheim",
    "12": "Bloodhound", "13": "Bluetick", "14": "Border Collie", "15": "Borzoi",
    "16": "Boston Terrier", "17": "Boxer", "18": "Bull Mastiff", "19": "Bull Terrier",
    "20": "Bulldog", "21": "Cairn", "22": "Chihuahua", "23": "Chinese Crested",
    "24": "Chow", "25": "Clumber", "26": "Cockapoo", "27": "Cocker", "28": "Collie",
    "29": "Corgi", "30": "Coyote", "31": "Dalmation", "32": "Dhole", "33": "Dingo",
    "34": "Doberman", "35": "Elk Hound", "36": "French Bulldog", "37": "German Sheperd",
    "38": "Golden Retriever", "39": "Great Dane", "40": "Great Perenees", "41": "Greyhound",
    "42": "Groenendael", "43": "Irish Spaniel", "44": "Irish Wolfhound", "45": "Japanese Spaniel",
    "46": "Komondor", "47": "Labradoodle", "48": "Labrador", "49": "Lhasa", "50": "Malinois",
    "51": "Maltese", "52": "Mex Hairless", "53": "Newfoundland", "54": "Pekinese", 
    "55": "Pit Bull", "56": "Pomeranian", "57": "Poodle", "58": "Pug", "59": "Rhodesian",
    "60": "Rottweiler", "61": "Saint Bernard", "62": "Schnauzer", "63": "Scotch Terrier",
    "64": "Shar_Pei", "65": "Shiba Inu", "66": "Shih-Tzu", "67": "Siberian Husky", 
    "68": "Vizsla", "69": "Yorkie"
}

def preprocess_image(image: UploadFile) -> np.ndarray:
    img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    img = ImageOps.fit(img, (224, 224), method=Image.BICUBIC)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Preprocess the image
        img_array = preprocess_image(file)

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        breed_name = class_names[str(predicted_class)]
        probability = predictions[0][predicted_class]

        return {"breed": breed_name, "probability": float(probability)}
    except Exception as e:
        return {"error": str(e)}

# Add a simple GET method
@app.get("/")
async def root():
    return {"message": "Dog breed classification API !"}
