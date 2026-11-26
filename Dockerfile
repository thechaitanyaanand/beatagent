FROM python:3.12-slim
# Debian's ffmpeg is built with librubberband (used for time-stretching)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps basic-pitch \
 && playwright install --with-deps chromium
# fetch the Beat This! checkpoint at build time, not on first request
RUN python -c "from beat_this.inference import File2Beats; File2Beats(device='cpu')"
COPY . .
ENV HOST=0.0.0.0 PYTHONIOENCODING=utf-8
EXPOSE 8000
CMD ["python", "run.py"]
