# Use official PyTorch CPU image as base
FROM pytorch/pytorch:2.8.0-cpu

# Set working directory inside container
WORKDIR /app

# Copy requirements and install necessary Python packages (excluding torch)
COPY requirements.txt .

# Install dependencies, excluding torch (already in base image)
RUN pip install --no-cache-dir --upgrade -r requirements.txt --no-deps

# Copy the rest of your app code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start FastAPI app
CMD ["uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
