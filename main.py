import os
from google import genai
from google.genai import types
from control4_tool import control_home_lighting
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define the tool for Gemini
home_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="control_home_lighting",
            description="Turns lights on/off or sets brightness. Use device_id 105 for Kitchen.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "device_id": types.Schema(type="INTEGER"),
                    "brightness": types.Schema(type="INTEGER")
                }
            )
        )
    ]
)

# This is what happens after Porcupine hears "Jarvis"
def ask_gemini(user_voice_command):
    response = client.models.generate_content(
        model="gemini-1.5-flash", # Use the flash model for speed
        contents=user_voice_command,
        config=types.GenerateContentConfig(
            tools=[home_tool]
        )
    )

    # Check if Gemini wants to call your function
    for part in response.candidates[0].content.parts:
        if part.function_call:
            # Execute the Control4 command!
            result = control_home_lighting(
                part.function_call.args["device_id"], 
                part.function_call.args["brightness"]
            )
            print(result)