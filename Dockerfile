# Dockerfile

# 1. Start with a lean, official Python 3.10 base image
FROM python:3.10-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy and install dependencies from your requirements.txt
COPY requirements.txt .
# Add the API-specific libraries to the requirements and install everything
RUN echo "fastapi" >> requirements.txt && \
    echo "uvicorn[standard]" >> requirements.txt && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# 4. Copy your application code and your model artifact into the container
COPY ./api.py .
COPY ./artifacts/model.joblib .

# 5. Expose port 8000 to allow traffic to the container
EXPOSE 8000

# 6. Define the command to run when the container starts
# This runs your actual application with detailed logging enabled.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
