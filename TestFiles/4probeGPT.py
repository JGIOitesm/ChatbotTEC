from transformers import pipeline, AutoTokenizer, Conversation

tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b', trust_remote_code=True)

chatbot = pipeline("conversational", "stabilityai/stablelm-2-zephyr-1_6b",trust_remote_code=True,tokenizer=tokenizer)

conversation = Conversation([{"role": "user","content": "Yo soy don Francisco, hijo de Pepe Ramirez quien recide en Ocampo. Yo soy también de la familia Guzmán."}])
conversation.add_message({"role": "user", "content": "¿Puede proporcionar mi nombre completo por favor?"})
conversation = chatbot(conversation, pad_token_id=tokenizer.eos_token_id)
print(conversation.messages[-1]["content"])