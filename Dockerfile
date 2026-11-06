FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY --chown=user isp ./isp
COPY --chown=user app.py ./
COPY --chown=user weights ./weights
COPY --chown=user examples ./examples
USER user
ENV GRADIO_TEMP_DIR=/tmp/gradio PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py"]
