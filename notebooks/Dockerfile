# using cuda 12.1.1, that my GPU supports
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04


# Install Python, pip, and packages
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3 python3-pip curl net-tools && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy everything else into container
COPY . .

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab instance on port 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
