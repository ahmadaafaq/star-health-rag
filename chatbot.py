from rag import ask

print("=" * 50)
print("  Star Health Insurance Chatbot")
print("=" * 50)
print("Type 'quit' or 'exit' to stop\n")

while True:
    question = input("You: ").strip()
    
    if not question:
        continue
    
    if question.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break
    
    print("\nAssistant: Searching...")
    answer = ask(question)
    print(f"\nAssistant: {answer}")
    print("-" * 50 + "\n")