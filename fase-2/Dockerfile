FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY train.py .
COPY models/ models/

RUN mkdir -p data output

CMD ["/bin/bash"]