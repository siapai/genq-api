from fastapi import FastAPI, Path, Query, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from starlette import status
import shutil
from keras.models import load_model
import classifier
import time as t
import onnxruntime


app = FastAPI()

origins = [
    "http://localhost:3000",
]

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

onnx_model_path = 'models/inception_v3_categorical_epoch_20.onnx'
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])


class GenderClassifier:
    name: str
    label: str
    confidence: float
    latency: float

    def __init__(self, name: str, label: str, confidence: str, latency: float) -> None:
        self.name: str = name
        self.label: str = label
        self.confidence: str = confidence
        self.latency: float = latency


class PredictionRequest(BaseModel):
    filename: str = Field(min_length=5)

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "1293912839123.jpg",
            }
        }


app.mount("/static", StaticFiles(directory="files/cropped"), name="static")


@app.post("/upload/")
async def create_upload_file(uploaded_file: UploadFile):
    file_ext = uploaded_file.filename.split(".")[-1]
    filename = f"{round(t.time() * 1000)}.{file_ext}"
    path = f"files/original/{filename}"
    with open(path, 'w+b') as file:
        shutil.copyfileobj(uploaded_file.file, file)

    return {
        'file': filename,
        'content': uploaded_file.content_type,
        'path': f"static/{filename}",
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(prediction_request: PredictionRequest):
    filenames = classifier.crop_faces_mtcnn(prediction_request.filename)

    results = []
    for filename in filenames:
        label, confidence, latency = classifier.predict(filename, session)
        results.append(
            GenderClassifier(
                name=filename,
                label=label,
                confidence=confidence,
                latency=latency
            )
        )
    return results
