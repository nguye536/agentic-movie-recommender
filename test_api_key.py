import os
import ollama

api_key = os.environ.get('OLLAMA_API_KEY')
print(f"API Key: {api_key[:20]}...")

client = ollama.Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {api_key}"},
)

print("\n1. Testing chat endpoint...")
try:
    resp = client.chat(
        model="gemma4:31b-cloud",
        messages=[{"role": "user", "content": "say hello"}],
    )
    print("✓ Chat endpoint works!")
except Exception as e:
    print(f"✗ Chat failed: {e}")

print("\n2. Testing embed endpoint...")
try:
    resp = client.embed(
        model="nomic-embed-text",
        input=["test movie description"],
    )
    print("✓ Embed endpoint works!")
    print(f"  Embedding shape: {len(resp['embeddings'])}")
except Exception as e:
    print(f"✗ Embed failed: {e}")
