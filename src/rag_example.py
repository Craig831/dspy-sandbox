from dotenv import load_dotenv
import os
import dspy
from dspy.utils import download
import mlflow
import ujson

# Load environment variables from .env file
load_dotenv()

# Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")

with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]


    
sonnet_lm = dspy.LM('anthropic/claude-3-sonnet-20240229', api_key=os.getenv('ANTHROPIC-API-KEY'))
dspy.configure(lm=sonnet_lm)

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()


quest_answer = dspy.ChainOfThought('question -> response')
response = quest_answer(question="should curly braces appear on their own line?")

print(response.response)
