
FROM python:3.9

#
WORKDIR /app

#
COPY . ./

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install -e app/

RUN pip install transformers

RUN pip install fastapi

RUN pip install 'uvicorn[standard]'

RUN pip install librosa

RUN pip install ipywidgets

RUN pip install python-multipart

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
