# ML Service for Image Classification

This service provides image classification capabilities for the Civic Lens application.

## Overview

The ML service uses a ResNet50 model trained to classify civic issues into 5 categories:
- Broken/Missing Streetlight
- Garbages
- Others
- Potholes
- Waterlogging

## Requirements

- Python 3.8+
- TensorFlow 2.13.0
- Flask 2.3.2
- Pillow 10.0.0
- NumPy 1.24.3

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the service:
```bash
python predict.py
```

The service will start on `http://localhost:5001`

## API Endpoints

### POST /predict
Classify an image and return the category.

**Request:**
- Form data with an 'image' field containing the image file

**Response:**
```json
{
  "category": "streetlight",
  "className": "Broken Missing Streetlight",
  "confidence": 0.95,
  "classIndex": 0
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "ML service is running"
}
```

## Integration with Backend

The Node.js backend sends images to this service when a user submits a complaint without selecting a category. The service returns the predicted category which is then used to create the complaint in the database.