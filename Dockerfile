# Dockerfile

# 1. Start with a lean, official Python 3.10 base image
FROM python:3.10-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy and install dependencies from your requirements.txt
# This is done first to take advantage of Docker's layer caching
COPY requirements.txt .
# Add the API-specific libraries to the requirements and install everything
RUN echo "fastapi" >> requirements.txt && \
    echo "uvicorn[standard]" >> requirements.txt && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# 4. Copy your application code and your model artifact into the container
# --- CHANGE THIS SECTION ---
COPY ./api.py .
# Copy the model directly from the local 'artifacts' folder into the container's '/app' directory
COPY ./artifacts/model.joblib .
# --- END OF CHANGES ---

# 5. Expose port 8000 to allow traffic to the container
EXPOSE 8000

# 6. Define the command to run when the container starts
# --host 0.0.0.0 is essential to make the API accessible outside the container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
