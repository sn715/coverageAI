import openai

class Generator:
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens
    
    def generate(self, prompt, context):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens
        )
        return response.choices[0].message['content']