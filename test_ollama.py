import ollama

print("Testing Ollama models...")

# Test LLaMA
try:
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Say 'LLaMA working' in 3 words"}]
    )
    print(f"✅ LLaMA 3.2: {response['message']['content']}")
except Exception as e:
    print(f"❌ LLaMA 3.2 not available: {e}")
    print("   Run: ollama pull llama3.2")

# Test GEMMA
try:
    response = ollama.chat(
        model="gemma2:2b",
        messages=[{"role": "user", "content": "Say 'GEMMA working' in 3 words"}]
    )
    print(f"✅ GEMMA 2: {response['message']['content']}")
except Exception as e:
    print(f"❌ GEMMA 2 not available: {e}")
    print("   Run: ollama pull gemma2:2b")