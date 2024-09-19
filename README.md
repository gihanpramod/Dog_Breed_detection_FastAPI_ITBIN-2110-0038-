# Dog Breed Classification API
This project provides a REST API built using FastAPI for classifying dog breeds from images. It uses a pre-trained TensorFlow model to predict the breed of a dog based on an uploaded image.

# Features
    *POST /predict: Upload an image and receive the predicted dog breed and probability.
    *GET /: A simple endpoint to check the status of the API.
# Requirements

To run this project, you'll need the following dependencies:

    *Python 3.8+
    *FastAPI
    *Uvicorn (ASGI server for FastAPI)
    *TensorFlow (for loading and using the dog breed classification model)
    *Pillow (for image processing)
    *Numpy (for array manipulations)
    *Matplotlib (optional, if you want to include visualization features)

# Install Dependencies

    - pip install fastapi uvicorn tensorflow pillow numpy python-multipart

# Model
The model used for this project is a pre-trained dog breed classification model. It classifies dog images into one of 70 breeds, which are mapped in the API to their respective breed names.

The model file (dogclassification.h5) should be placed in the project directory. If you don't have this file, you need to train or obtain a TensorFlow model that classifies dog breeds.

# How to Run
Clone the repository to your local machine.

Install the required dependencies:
    -pip install -r requirements.txt

Run the FastAPI application using Uvicorn:
    -uvicorn main:app --reload

The API will be running at:
http://127.0.0.1:8000

# API Endpoints
# POST /predict
This endpoint accepts an image file and returns the predicted breed and the probability score.

    *URL: /predict
    *Method: POST
    *Body: Upload a file (image)
    
    Response:
{
  "breed": "Labrador",
  "probability": 0.95
}


Example with curl:

curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@dog_image.jpg'

# GET /
This is a simple endpoint that returns a message indicating the API is running.

    *URL: /
    *Method: GET

Response:

{
  "message": "Dog breed classification API is running!"
}

# Folder Structure


.
├── dogclassification.h5        # The trained dog breed classification model
├── main.py                     # FastAPI application
├── README.md                   # This README file
├── requirements.txt            # Dependencies for the project