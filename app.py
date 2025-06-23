from flask import Flask, request, jsonify
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def query_openrouter_deepseek(messages, model="deepseek/deepseek-chat"):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def run_agent(user_message):
    messages = [
        {"role": "system", "content": "You are a helpful Discord bot AI agent."},
        {"role": "user", "content": user_message}
    ]
    return query_openrouter_deepseek(messages)

def detect_intent(message):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent detection assistant. "
                "Your job is to extract the user's intent from a message. "
                "Possible intents: 'sound', 'music', 'nasa', 'none'. "
                "If the intent is 'music', extract the URL if one is present. "
                "Return your answer ONLY as a JSON object using this format:\n\n"
                "{ \"intent\": \"intent_name\", \"url\": \"optional_url_or_null\" }\n\n"
                "Examples:\n"
                "User: 'show me a pic from space'\n"
                "→ { \"intent\": \"nasa\", \"url\": null }\n"
                "User: 'make it loud'\n"
                "→ { \"intent\": \"sound\", \"url\": null }\n"
                "User: 'play https://youtube.com/xyz song please'\n"
                "→ { \"intent\": \"music\", \"url\": \"https://youtube.com/xyz\" }\n"
                "User: 'how are you?'\n"
                "→ { \"intent\": \"none\", \"url\": null }\n"
                "Do NOT wrap your output in code blocks. Output only the text, raw and unformatted."
            )
        },
        {
            "role": "user",
            "content": message
        }
    ]

    response = query_openrouter_deepseek(messages)
    
    if response.startswith("```json") or response.startswith("```"):
        response = response.strip('`')
        if response.startswith("json"):
            response = response[4:].strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Failed to parse LLM response:", response)
        return {"intent": "none", "url": None}

@app.route("/agent", methods=["POST"])
def agent_endpoint():
    data = request.json
    user_message = data.get("message", "")

    intent_data = detect_intent(user_message)
    intent = intent_data.get("intent", "none")
    url = intent_data.get("url", None)

    if intent == "sound":
        return jsonify({"intent": "sound", "response": None})
    elif intent == "music":
        return jsonify({"intent": "music", "response": None, "url": url})
    elif intent == "nasa":
        return jsonify({"intent": "nasa", "response" : None})

    reply = run_agent(user_message)
    return jsonify({"intent": "none", "response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
