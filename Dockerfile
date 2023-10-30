#FROM tiangolo/uvicorn-gunicorn:python3.11-slim
FROM python:3.11

RUN apt-get update && apt-get install -y netcat-traditional && apt-get install -y --no-install-recommends git

WORKDIR /app

COPY . ./

RUN pip install -e app/

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install transformers

RUN pip install fastapi

RUN pip install 'uvicorn[standard]'

RUN pip install librosa

RUN pip install ipywidgets

RUN pip install python-multipart

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "80"]