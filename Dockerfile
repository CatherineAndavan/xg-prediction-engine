# 1. Using a lightweight Python image
FROM python:3.10-slim

# 2. Setting the working directory inside the container
WORKDIR /app

# 3. Copying the requirements file first 
COPY requirements.txt .

# 4. Installing the necessary football and ML libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copying the rest of the project files
COPY app.py src .

# 6. Exposing the port FastAPI will run on
EXPOSE 8000

# 7. Command to run the API
# We use 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
