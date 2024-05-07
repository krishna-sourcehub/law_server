from huggingface_hub import hf_hub_download

# Define the repository ID (username/repository_name)
repo_id = 'TheBloke/Llama-2-7B-Chat-GGUF'

# Define the filename you want to download
filename = 'llama-2-7b-chat.Q2_K.gguf'

# Define the cache directory (optional)
# If not provided, the default cache directory will be used
cache_dir = './path_to_cache_directory'

def model_download():
    # Download the file
    file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    cache_dir=cache_dir)   # The file_path variable now contains the local path to the downloaded file
    print(f"File downloaded to: {file_path}")
