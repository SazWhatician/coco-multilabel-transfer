## Setup & Run

NOTE: The given Dataset for this project was unavailable on Kaggle - "https://www.kaggle.com/datasets/shubham2703/coco-dataset-for-multi-label-image-classification/data"

Hence, The Pascal VOC Dataset is used instead for the completion of the project.

1. Install dependencies

pip install -r requirements.txt

2. Set Groq API key

$env:GROQ_API_KEY=""  # PowerShell

export GROQ_API_KEY=""  # Linux/Mac



3. Start the server
uvicorn app:app --reload

4. Test via Swagger UI
http://127.0.0.1:8000/docs

   OR run test_api.ipynb
