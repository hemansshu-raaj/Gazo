# The "Take a Pic" website (FastAPI + static site), CPU only.
#   docker build -t take-a-pic . && docker run --rm -p 8000:8000 take-a-pic   ->  http://localhost:8000
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY web/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user isp ./isp
COPY --chown=user web ./web
COPY --chown=user weights ./weights
COPY --chown=user examples ./examples
USER user
ENV PYTHONUNBUFFERED=1 ISP_THREADS=4
EXPOSE 8000
CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
