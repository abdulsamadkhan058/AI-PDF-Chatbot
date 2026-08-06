import asyncio
import edge_tts
VOICE = "en-US-AriaNeural"
async def generate_voice(text):
    communicate = edge_tts.Communicate(text,VOICE)
    await communicate.save("output.mp3")

def speak_text(text):
    asyncio.run(generate_voice(text))