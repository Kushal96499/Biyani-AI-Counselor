import requests
import json

def chat():
    welcome_msg = """
Dear student,
Welcome to Biyani AI Assistant. This AI chatbot has been developed after 20 years of teaching experience of team. It will help you with correct answers for all the complex questions of academic subjects. However, consult your teacher before any final decision.
-Team Biyani.
    """
    print(welcome_msg)
    print("\n--- (Type 'exit' to quit) ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
            
        url = "http://127.0.0.1:8000/chat"
        payload = {"message": query}
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"\nAI: {data['answer']}")
                if data['sources']:
                    print(f"Sources: {', '.join(data['sources'])}")
            else:
                print(f"\nError: {response.status_code}")
        except Exception as e:
            print(f"\nFailed to connect: {e}")

if __name__ == "__main__":
    chat()
