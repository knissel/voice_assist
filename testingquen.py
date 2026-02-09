import requests

# List of prompts you want to try
prompts = [
    "This is Jay Nissel and I love Bo Nissel and Kenny Nissel. I am amazing at pickleball!"
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