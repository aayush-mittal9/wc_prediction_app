# app/Dockerfile

FROM tensorflow/tensorflow:latest
 
EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

CMD ["bash"]
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]