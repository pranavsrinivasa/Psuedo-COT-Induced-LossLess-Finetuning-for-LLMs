from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import CoTLLM

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models and tokenizer
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the models
model_finetuned = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(torch_device)
model_finetuned.load_adapter('backend\Main_Adapters')

TTM_id = 'unsloth/Llama-3.2-1B-Instruct-bnb-4bit'
TTM_model = AutoModelForCausalLM.from_pretrained(
    TTM_id,
    torch_dtype=torch.bfloat16
).to(torch_device)
TTM_model.load_adapter('backend\TTM_Adapters')

tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = CoTLLM(model_finetuned,TTM_model,tokenizer)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response using your model
        response = llm.generate_output(user_message)
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)