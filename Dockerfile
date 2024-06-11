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
    libice6 \
    git \
    curl

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# Copy the rest of the application
COPY . .

# Install Python dependencies
RUN pip install flask tensorflow langdetect pytesseract opencv-python-headless pillow scipy transformers tensorrt

# Expose the application port
EXPOSE 5000

# Set the command to run the application
CMD ["python", "app.py"]
