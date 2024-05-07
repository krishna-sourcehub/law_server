# from flask import Flask, request, jsonify
# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download
# from model import model_download
# # model_download()

# # Initialize the Llama model with chat format set to "llama-2"
# llm = Llama(model_path="./path_to_cache_directory/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q2_K.gguf", chat_format="llama-2")

# # Define the system prompt
# system_prompt = (
#     "[INSTRUCTION] You are a chatbot named 'Makkal Thunaivan' designed to provide legal support to marginalized communities in India. "
#     "You were fine-tuned by Sathish Kumar and his team members at the University College of Engineering Dindigul. "
#      "Developer Team members include Karthikeyan as Model Trainer, Prashanna as Dataset Researcher, Nivas as Model Architect, and Sathish Kumar as Team Leader and Frontend Developer and Model Tester. "
#     "Your purpose is to answer questions related to Indian law and marginalized communities in India. "
#     "You have been trained on various legal topics. "
#     "Your responses should be concise, meaningful, and accurate."
#     "When a user asks for more information or details, provide a more comprehensive explanation. "
#     "Your responses should be respectful and informative."
#     "Do not provide information unrelated to India or Indian law. "
#     "Feel free to ask questions."
# )

# # Initialize the conversation history list with the system prompt
# conversation_history = [{"role": "system", "content": system_prompt}]

# # Create a Flask application
# app = Flask(__name__)

# # Define the model function
# def model(query):
#     global conversation_history  # Declare global to update history

#     # Add the user's query to the conversation history
#     conversation_history.append({"role": "user", "content": query})

#     # Calculate the total number of tokens in the conversation history
#     # (You may need to modify this part to calculate the token count accurately based on your tokenizer)
#     total_tokens = sum(len(message["content"].split()) for message in conversation_history)

#     # If the total number of tokens exceeds the model's context window, trim the history
#     # You may need to adjust the 512 value based on your model's actual context window size
#     context_window_size = 512
#     while total_tokens > context_window_size:
#         # Remove the oldest messages from the conversation history
#         conversation_history.pop(0)
#         # Recalculate the total number of tokens
#         total_tokens = sum(len(message["content"].split()) for message in conversation_history)

#     # Generate chat completion with the conversation history
#     response = llm.create_chat_completion(messages=conversation_history, max_tokens=75)
    
#     # Extract the assistant's response from the completion dictionary
#     if response and 'choices' in response and response['choices']:
#         assistant_response = response['choices'][0]['message']['content']
#         assistant_response = assistant_response.strip()
        
#         # Add the assistant's response to the conversation history
#         conversation_history.append({"role": "assistant", "content": assistant_response})

#         # Print the assistant's response
#         print("Assistant response:", assistant_response)
        
#         # Return the assistant's response
#         return assistant_response
#     else:
#         print("Error: Invalid response structure.")
#         return None


# # Define the endpoint for the API
# @app.route("/chat", methods=["GET"])
# def chat_endpoint():
#     # Get the query parameter from the request
#     query = request.args.get("query")
    
#     # Check if the "refresh" parameter is set to "true"
#     refresh = request.args.get("refresh")
#     if refresh and refresh.lower() == "true":
#         # Clear the conversation history
#         global conversation_history
#         conversation_history = [{"role": "system", "content": system_prompt}]
#         return jsonify({"response": "Conversation history cleared."})
    
#     # If there is no query, return an error message
#     if not query:
#         return jsonify({"error": "Query parameter is required."}), 400
    
#     # Call the model function with the query
#     response = model(query)
    
#     # Return the assistant's response as JSON
#     return jsonify({"response": response})

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)



from flask import Flask, request, jsonify
from llama_cpp import Llama
import logging
from model import model_download
model_download()
# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the Llama model with chat format set to "llama-2"
llm = Llama(model_path="./llama-2-7b-chat.Q2_K.gguf", chat_format="llama-2")

# Define the system prompt
system_prompt = (
    "[INSTRUCTION] You are a chatbot named 'Makkal Thunaivan' designed to provide legal support to marginalized communities in India. "
    "You were fine-tuned by Sathish Kumar and his team members at the University College of Engineering Dindigul. "
     "Developer Team members include Karthikeyan as Model Trainer, Prashanna as Dataset Researcher, Nivas as Model Architect, and Sathish Kumar as Team Leader and Frontend Developer and Model Tester. "
    "Your purpose is to answer questions related to Indian law and marginalized communities in India. "
    "You have been trained on various legal topics. "
    "Your responses should be concise, meaningful, and accurate."
    "When a user asks for more information or details, provide a more comprehensive explanation. "
    "Your responses should be respectful and informative."
    "Do not provide information unrelated to India or Indian law. "
    "Feel free to ask questions."
)

# Initialize the conversation history list with the system prompt
conversation_history = [{"role": "system", "content": system_prompt}]

# Define conversation history size limit
MAX_CONVERSATION_HISTORY_SIZE = 10

# Create a Flask application
app = Flask(__name__)

# Define a function to calculate the total number of tokens in conversation history using the Llama model's tokenizer
def calculate_total_tokens(messages):
    try:
        # Convert content to string and tokenize
        total_tokens = sum(len(llm.tokenize(str(message["content"]), add_bos=False, special=True)) for message in messages)
        return total_tokens
    except Exception as e:
        logging.error(f"Error during tokenization: {e}")
        return 0  # Return a safe value (0) to handle the error

# Define a function to trim the conversation history if the total number of tokens exceeds the context window size
def trim_conversation_history():
    global conversation_history
    total_tokens = calculate_total_tokens(conversation_history)
    context_window_size = 512

    while total_tokens > context_window_size:
        # Remove the oldest messages from the conversation history
        conversation_history.pop(0)
        # Recalculate the total number of tokens
        total_tokens = calculate_total_tokens(conversation_history)

# Define the model function
def model(query):
    global conversation_history

    # Add the user's query to the conversation history
    conversation_history.append({"role": "user", "content": query})

    # Calculate the total number of tokens in the conversation history
    total_tokens = calculate_total_tokens(conversation_history)

    # If the total number of tokens exceeds the model's context window, trim the history
    trim_conversation_history()

    # Generate chat completion with the conversation history
    try:
        response = llm.create_chat_completion(messages=conversation_history, max_tokens=200)

        # Extract the assistant's response from the completion dictionary
        if response and 'choices' in response and response['choices']:
            assistant_response = response['choices'][0]['message']['content']
            assistant_response = assistant_response.strip()

            # Add the assistant's response to the conversation history
            conversation_history.append({"role": "assistant", "content": assistant_response})

            # Return the assistant's response
            return assistant_response
        else:
            logging.error("Error: Invalid response structure.")
            return None
    except Exception as e:
        logging.error(f"Error during chat completion: {e}")
        return None

# Define the endpoint for the API
@app.route("/chat", methods=["GET"])
def chat_endpoint():
    # Get the query parameter from the request
    query = request.args.get("query")

    # Check if the "refresh" parameter is set to "true"
    refresh = request.args.get("refresh")
    if refresh and refresh.lower() == "true":
        # Clear the conversation history
        global conversation_history
        conversation_history = [{"role": "system", "content": system_prompt}]
        return jsonify({"response": "Conversation history cleared."})

    # If there is no query, return an error message
    if not query:
        return jsonify({"error": "Query parameter is required."}), 400

    # Call the model function with the query
    response = model(query)

    # Return the assistant's response as JSON
    if response is None:
        return jsonify({"error": "An error occurred while processing the request."}), 500

    # Check the size of the conversation history and clear if necessary
    if len(conversation_history) > MAX_CONVERSATION_HISTORY_SIZE:
        conversation_history = [{"role": "system", "content": system_prompt}]
        return jsonify({"response": response, "notification": "Conversation history was cleared due to exceeding maximum size."})
    print(response)
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
