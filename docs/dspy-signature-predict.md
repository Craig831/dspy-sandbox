[DSPy Signature Docs](https://dspy.ai/learn/programming/signatures)

```python
from dotenv import load_dotenv
import os
import dspy
import mlflow

# Load environment variables from .env file
load_dotenv()

sonnet_lm = dspy.LM('anthropic/claude-3-sonnet-20240229', api_key=os.getenv('ANTHROPIC-API-KEY'))
dspy.configure(lm=sonnet_lm)

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()

# quest_answer is a simple example of a DSPy signature
quest_answer = dspy.Predict('question: str -> response: str')
response = quest_answer(question="what are high memory and low memory on linux?")

print(response.response)
```