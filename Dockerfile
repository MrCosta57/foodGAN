FROM python:3.11-slim

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir gdown

WORKDIR /app

RUN gdown --folder https://drive.google.com/drive/folders/1MolNVjt3HZZ3bihJG1Cf0_Ij_ChbOE2Q?usp=sharing -O checkpoints

COPY dataset/labels.csv dataset/labels.csv
COPY ./src src

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
