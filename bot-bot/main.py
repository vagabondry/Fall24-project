import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
import json
from datetime import datetime
import os
import random
import hashlib

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
special_tokens_dict = {'additional_special_tokens': ['[SEP]']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
base_model.resize_token_embeddings(len(tokenizer))

adapter_path = "lora"
lora_config = PeftConfig.from_pretrained(adapter_path)

model = PeftModel.from_pretrained(base_model, adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

feedback_context = {}

def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.to(device)

    output = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        num_beams=5,  # Beam search for variety
        repetition_penalty=2.0,  # Penalize repetition
        no_repeat_ngram_size=3,  # Prevent repetition of phrases
        top_k=50,  # Use top-k sampling to reduce randomness
        top_p=0.9,  # Use nucleus sampling for diversity
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def log_message(message_data, chat_id):
    try:
        os.makedirs("chat_logs", exist_ok=True)
        file_name = f"chat_logs/chat_{chat_id}.json"

        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                logs = json.load(file)
        else:
            logs = []

        logs.append(message_data)

        with open(file_name, "w") as file:
            json.dump(logs, file, indent=4)
    except Exception as e:
        print(f"Error logging message for chat {chat_id}: {e}")

def log_feedback(feedback_data, chat_id):
    try:
        os.makedirs("feedback_logs", exist_ok=True)
        file_name = f"feedback_logs/chat_{chat_id}_feedback.json"

        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                logs = json.load(file)
        else:
            logs = []

        logs.append(feedback_data)

        with open(file_name, "w") as file:
            json.dump(logs, file, indent=4)
    except Exception as e:
        print(f"Error logging feedback for chat {chat_id}: {e}")

async def start(update: Update, context):
    if update.message.chat.type == "private":
        await update.message.reply_text("Hi! I'm your AI bot. Send me a message!")
    else:
        await update.message.reply_text("Hi! Add me to your group and mention me to interact.")

async def chat(update: Update, context):
    user_message = update.message.text
    chat_id = update.message.chat.id
    chat_type = update.message.chat.type
    bot_username = context.bot.username

    message_data = {
        "chat_id": chat_id,
        "user_id": update.message.from_user.id,
        "username": update.message.from_user.username,
        "message": user_message,
        "timestamp": datetime.now().isoformat(),
        "chat_type": chat_type,
    }
    log_message(message_data, chat_id)

    if chat_type in ["group", "supergroup"]:
        if f"@{bot_username}" in user_message:
            user_message = user_message.replace(f"@{bot_username}", "").strip()
        else:
            if random.random() > 0.33:
                return

    try:
        response = generate_response(user_message, model, tokenizer)

        feedback_id = hashlib.md5(f"{chat_id}:{user_message}:{response}".encode()).hexdigest()
        feedback_context[feedback_id] = {
            "user_message": user_message,
            "bot_response": response,
            "chat_id": chat_id,
            "timestamp": datetime.now().isoformat(),
        }

        keyboard = [
            [
                InlineKeyboardButton("Good 👍", callback_data=f"feedback:Good:{feedback_id}"),
                InlineKeyboardButton("Bad 👎", callback_data=f"feedback:Bad:{feedback_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(response, reply_markup=reply_markup)

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {e}")

async def feedback_handler(update: Update, context):
    query = update.callback_query
    await query.answer()

    data = query.data.split(":")
    feedback = data[1]  # Good or Bad
    feedback_id = data[2]  # Feedback ID
    feedback_data = feedback_context.get(feedback_id)

    if feedback_data:
        feedback_entry = {
            "user_id": query.from_user.id,
            "username": query.from_user.username,
            "chat_id": feedback_data["chat_id"],
            "user_message": feedback_data["user_message"],
            "bot_response": feedback_data["bot_response"],
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        log_feedback(feedback_entry, feedback_data["chat_id"])

        await query.edit_message_reply_markup(reply_markup=None)

BOT_TOKEN = "7757297121:AAE-1Ny0XVNkJrwfsO1uoT7_WMJT-v3J-2E"

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
app.add_handler(CallbackQueryHandler(feedback_handler))

if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
