import requests

# List of prompts you want to try
prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "I can say anything you want with high quality voice cloning!",
    "Testing the emotional range of the new Qwen three T T S model."
]

for i, text in enumerate(prompts):
    print(f"Generating audio for: {text}")
    response = requests.post(
        "http://localhost:5001/synthesize",
        json={"text": text, "language": "english"}
    )
    
    if response.status_code == 200:
        filename = f"prompt_{i}.wav"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f" Saved to {filename}")
    else:
        print(f" Error: {response.text}")