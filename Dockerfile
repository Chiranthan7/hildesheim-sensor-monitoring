# Hildesheim Smart Sensor Monitoring - web app
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8050
EXPOSE 8050
# the hosting service sets PORT; the app serves the prepared data in ./appdata
CMD python -m hsm.cli app --host 0.0.0.0 --port ${PORT} --data appdata
