import requests
import uuid
import os

url = "http://localhost:8000/api/v1/documents"
file_path = "d:/PROJECT/parsing/testingData/ar01.pdf"

user_id = str(uuid.uuid4())
subject_id = str(uuid.uuid4())

data = {
    "user_id": user_id,
    "subject_id": subject_id,
    "subject_name": "Test Subject From Python",
    "mode": "fast",
    "ocr_engine": "tesseract"
}

with open(file_path, "rb") as f:
    files = {"file": (os.path.basename(file_path), f, "application/pdf")}
    response = requests.post(url, data=data, files=files)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
print(f"User ID: {user_id}")
print(f"Subject ID: {subject_id}")
