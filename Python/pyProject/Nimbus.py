# Nimbus(chatBot)

# basic chatBot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

chatbot = ChatBot("ChatBot")

# Training chatBot
trainer = ListTrainer(chatbot)
trainer.train([
    "Hi",
    "Welcome 🤗",
])
trainer.train([
    "Are you a human?",
    "No, I'm Nova built using natural language processing and machine learning techniques!",
])

exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        print(f"🪴 {chatbot.get_response(query)}")




