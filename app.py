# Use a pipeline as a high-level helper
"""
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-405B-Instruct")
pipe(messages)
"""

# Load model directly
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
"""

# Datasets
"""
from datasets import load_dataset

ds = load_dataset("open-llm-leaderboard/meta-llama__Meta-Llama-3.1-8B-Instruct-details", "meta-llama__Meta-Llama-3.1-8B-Instruct__leaderboard_bbh_boolean_expressions")
ds = load_dataset("open-llm-leaderboard/meta-llama__Meta-Llama-3.1-8B-Instruct-details", "meta-llama__Meta-Llama-3.1-8B-Instruct__leaderboard_bbh_causal_judgement")
ds = load_dataset("open-llm-leaderboard/meta-llama__Meta-Llama-3.1-8B-Instruct-details", "meta-llama__Meta-Llama-3.1-8B-Instruct__leaderboard_bbh_date_understanding")

data = load_dataset(
    "open-llm-leaderboard/meta-llama__Meta-Llama-3.1-8B-Instruct-details-private",
    name="meta-llama__Meta-Llama-3.1-8B-Instruct__leaderboard_bbh_boolean_expressions",
    split="latest"
)


data = load_dataset(
    "meta-llama/Meta-Llama-3.1-8B-evals",
    name="Meta-Llama-3.1-8B-evals__agieval_english__details",
    split="latest"
    )

data = load_dataset(
    "meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
    name="Meta-Llama-3.1-8B-Instruct-evals__mbpp__details",
    split="latest"
    )

"""


# Model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
# Use a pipeline as a high-level helper
"""
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
pipe(messages)
"""

# Load model directly
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
model = AutoModelForCausalLM.from_pretrained("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")

"""



"""
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
	filename="Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf",
)

llm.create_chat_completion(
		messages = [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
)
"""

"""
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
"""

"""
# app.py
import gradio as gr

gr.load("models/meta-llama/Meta-Llama-3-8B").launch()
"""


"""
# Transformers library
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
pipe(messages)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
"""


"""
# app.py
 
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
 
# model_url = "https://huggingface.co/mradermacher/llama-3-8b-gpt-4o-GGUF/resolve/main/llama-3-8b-gpt-4o.f16.gguf"
model_url = "mradermacher/llama-3-8b-gpt-4o-GGUF"
 
try:

    tokenizer = AutoTokenizer.from_pretrained(model_url)
    model = AutoModel.from_pretrained(model_url)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
 

# Create generation function
def generate_text(prompt):
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating text: {e}"        

# Create Gradio interface
interface = gr.Interface(fn=generate_text, inputs="text", outputs="text")
interface.launch()

"""


"""
from datasets import load_dataset

ds = load_dataset("open-llm-leaderboard/meta-llama__Meta-Llama-3.1-8B-Instruct-details", "meta-llama__Meta-Llama-3.1-8B-Instruct__leaderboard_bbh_boolean_expressions")


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("meta-llama/Meta-Llama-3.1-8B-evals", "Meta-Llama-3.1-8B-evals__agieval_english__details")


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", "Meta-Llama-3.1-8B-Instruct-evals__api_bank__details")
"""

"""
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
pipe(messages)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
model = AutoModelForCausalLM.from_pretrained("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
"""

# from datasets import load_dataset

# Login using `huggingface-cli login` to access the dataset
# ds = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", "Meta-Llama-3.1-8B-Instruct-evals__api_bank__details")


# ds = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", "Meta-Llama-3.1-8B-Instruct-evals__arc_challenge__details")


"""
import gradio as gr

gr.load("models/meta-llama/Meta-Llama-3.1-8B-Instruct").launch()

"""



import gradio as gr

import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# from huggingface_hub import notebook_login
# notebook_login()

from huggingface_hub import login
from dotenv import load_dotenv

# login()

load_dotenv()
#  token = os.environ['sorsw']
token = 'TOKEN';
login(token)


dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", "Meta-Llama-3.1-8B-Instruct-evals__arc_challenge__details", use_auth_token=True)
 
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Training function (optional)
def train_model(epochs=3):
    training_args = TrainingArguments(
        output_dir="output",  # Adjust output directory
        per_device_train_batch_size=8,  # Adjust batch size
        num_train_epochs=epochs,
        evaluation_strategy="epoch",  # Adjust evaluation strategy
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    print("Model training complete!")

# Text generation function
def generate_text(prompt):
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating text: {e}"

# Gradio interface for text generation
interface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="Text Generation with Trained Model",
    description="Enter a prompt and get creative text generated by the model.",
)

# Train the model before launching the interface (optional)
train_model()  # Uncomment to train before launching

# Launch the Gradio interface
interface.launch()




