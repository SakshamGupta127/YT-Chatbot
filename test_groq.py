from groq import Groq

def test_groq_connection():
    """Test Groq API connection"""
    api_key = "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"
    
    try:
        client = Groq(api_key=api_key)
        
        # Test with a simple question
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Correct model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you confirm you're working?"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        print("✅ Groq API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_groq_connection()
