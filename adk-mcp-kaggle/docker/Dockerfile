FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080 8081
CMD bash -lc "uvicorn adk_app.app:app --host 0.0.0.0 --port 8080 & uvicorn mcp_server.server:app --host 0.0.0.0 --port 8081"
