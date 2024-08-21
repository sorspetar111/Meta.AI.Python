# FROM python:3.9

# RUN apt-get update && apt-get install -y wget

# RUN useradd -m -u 1000 user
# USER user
# ENV PATH="/home/user/.local/bin:$PATH"

# WORKDIR /app

# RUN pip install datasets

# RUN wget https://github.com/huggingface/datasets/archive/refs/heads/main.zip -O datasets.zip
# RUN unzip datasets.zip -d datasets

# COPY --chown=user requirements.txt requirements.txt
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

# COPY --chown=user . /app
# COPY --chown=user app.py /app/app.py

# CMD ["python", "app.py"]






# Dockerfile

FROM python:3.9

# Install required dependencies
RUN apt-get update && apt-get install -y wget

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

RUN pip install datasets
RUN pip install huggingface_hub
# RUN python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('TOKEN')"
# RUN pip install dotenv
RUN pip install load_dotenv 


# Clone the Hugging Face datasets library
# Install the datasets library
# Copy your Python script (e.g., `load_dataset.py`)

RUN wget https://github.com/huggingface/datasets/archive/refs/heads/main.zip -O datasets.zip
RUN unzip datasets.zip -d datasets

# RUN wget https://huggingface.co/datasets/datasets/archive/ref/main.zip -O datasets.zip --chown=user
# RUN unzip datasets.zip -d datasets --chown=user


# RUN rm datasets.zip
# WORKDIR datasets/datasets
# RUN pip install -e .
# COPY load_dataset.py .
# COPY app.py .


# Copy model files
#COPY --chown=user Meta-Llama-3.1-8B-Instruct /app/Meta-Llama-3.1-8B-Instruct

# Install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# Copy application code
COPY --chown=user app.py /app/app.py

CMD ["python", "app.py"]


# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
