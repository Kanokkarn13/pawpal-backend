from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from datetime import datetime

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pawpal_backend.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "./model"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_K = 50

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Optional: optimize CUDA usage
if device == "cuda":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# Load model with enhanced error handling
try:
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if device == "cuda":
        model = model.half()

    # Warm-up run
    with torch.no_grad():
        test_input = tokenizer.encode("Test", return_tensors="pt").to(device)
        model.generate(test_input, max_new_tokens=10)

    logging.info("Model loaded successfully!")

except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    if "CUDA out of memory" in str(e):
        logging.error("Try reducing model size or using CPU")
    exit(1)

@app.route('/api/chat', methods=['POST'])
def chat():
    start_time = datetime.now()
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()

        if not user_input:
            logging.warning("Empty message received")
            return jsonify({"error": "Message cannot be empty"}), 400

        logging.info(f"Processing query: '{user_input}'")

        inputs = tokenizer.encode(
            user_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=1,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        bot_response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        processing_time = (datetime.now() - start_time).total_seconds()

        if device == "cuda":
            logging.info(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logging.info(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        logging.info(f"Response generated in {processing_time:.2f}s")

        return jsonify({
            "response": bot_response,
            "processing_time": processing_time
        })

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        error_msg = "GPU memory exhausted. Try shorter inputs or switch to CPU mode."
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500

    except Exception as e:
        logging.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": device,
        "model_loaded": True
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
