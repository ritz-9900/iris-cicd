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
# --- ✅ TEMPORARY DEBUGGING STEP ---
# This command replaces the uvicorn server with a simple loop that prints to the log.
# This will test if the GKE logging system is capturing ANY output from the container.
CMD ["python", "-c", "import time; print('--- LOGGING TEST STARTED ---'); [print(f'Logging... {i}'); time.sleep(5) for i in range(120)]"]
