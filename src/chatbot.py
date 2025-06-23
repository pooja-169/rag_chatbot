# src/chatbot.py

from generator import generate_answer

def main():
    print("🤖 Welcome to the RAG Chatbot! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        answer = generate_answer(query)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main()
