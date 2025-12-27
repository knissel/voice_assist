import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("Available voices on your Mac:\n")
for i, voice in enumerate(voices):
    print(f"{i}. {voice.name}")
    print(f"   ID: {voice.id}\n")

# Test a voice
print("\nTesting voice...")
engine.say("Hello, I am your smart home assistant. How may I help you today?")
engine.runAndWait()
