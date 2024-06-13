# Use the official PyTorch image as a base to avoid dependency issues
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6

# Copy the rest of the application
COPY . .

# Install Python dependencies
RUN pip install flask tensorflow matplotlib langdetect pytesseract opencv-python-headless pillow scipy transformers=4.36.2 tensorrt PyYAML datasets seqeval imutils

# Expose the application port
EXPOSE 5000

# Set the command to run the application
CMD ["python", "app.py"]
