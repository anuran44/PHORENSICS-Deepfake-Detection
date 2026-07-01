# 1. Base Image: Use a lightweight version of Python 3.10
FROM python:3.10-slim

# 2. Set environment variables to keep Python behavior predictable in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. System Dependencies:
# - default-jre: Required by PySpark to run the JVM
# - procps: Required by psutil to read CPU/RAM telemetry
# - curl: Required for Docker health checks
RUN apt-get update && apt-get install -y \
    default-jre \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# 6. Install the Python dependencies (keeping the image slim by clearing cache)
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy your main application code into the container
COPY phorensics.py .

# 8. Expose the port Streamlit uses to communicate
EXPOSE 8501

# 9. Optional but professional: Add a healthcheck to ensure Streamlit booted successfully
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. The command that runs when someone boots the container
ENTRYPOINT ["streamlit", "run", "phorensics.py", "--server.port=8501", "--server.address=0.0.0.0"]