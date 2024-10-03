# chat_interface.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    chat_history_ids = None

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Goodbye!")
            break

        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        chat_history_ids = model.generate(  
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    main()
