from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API key not found")
    exit()

print("✅ API key mil gayi")

client = genai.Client(api_key=api_key)

interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input="Say hello in one short sentence."
)

print("✅ Gemini API connected!")
print("Gemini:", interaction.output_text)