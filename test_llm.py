from core.llm import load_llm

llm = load_llm()

response = llm.invoke(
    "Say hello in one short sentence."
)

print("✅ LLM working!")
print(response.content)