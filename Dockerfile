FROM python:3.9

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

RUN wget https://github.com/huggingface/datasets/archive/refs/heads/main.zip -O datasets.zip
RUN unzip datasets.zip -d datasets

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
COPY --chown=user app.py /app/app.py

CMD ["python", "app.py"]

