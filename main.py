from fastapi import FastAPI, Path, Query, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from starlette import status
import shutil
from keras.models import load_model
import classifier
import time as t

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

model = load_model('models/inception_v3_categorical_epoch_20.keras')


class GenderClassifier:
    name: str
    result: str
    score: str
    time: float

    def __init__(self, name: str, result: str, score: str, time: float) -> None:
        self.name: str = name
        self.result: str = result
        self.score: str = score
        self.time: float = time


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


@app.post("/predict/", status_code=status.HTTP_200_OK)
async def predict(prediction_request: PredictionRequest):
    gender, score, inference_time = classifier.predict(prediction_request.filename, model)
    return GenderClassifier(
        name=prediction_request.filename,
        result=gender,
        score=score,
        time=inference_time
    )


@app.post("/predict/crop", status_code=status.HTTP_200_OK)
async def predict(prediction_request: PredictionRequest):
    filenames = classifier.crop_faces(prediction_request.filename)

    results = []
    for filename in filenames:
        gender, score, inference_time = classifier.predict(filename, model)
        results.append(
            GenderClassifier(
                name=filename,
                result=gender,
                score=score,
                time=inference_time
            )
        )

    return results
