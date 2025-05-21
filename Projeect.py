# pip install transformers gradio matplotlib emoji torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import emoji

# 📥 Load 3-Class Pretrained Sentiment Model (Twitter-RoBERTa)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 📊 Sentiment Prediction + Class Probabilities Visualization Function
def predict_sentiment(text):
    # Convert emojis to text description (e.g. 🤩 -> :star_struck:)
    text = emoji.demojize(text)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128, return_attention_mask=True)

    # Forward pass to get logits
    outputs = model(**inputs)
    logits = outputs.logits

    # Prediction
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    labels = ["Negative", "Neutral", "Positive"]
    class_id = np.argmax(probs)
    confidence = probs[class_id]

    # Plot class probabilities
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs, color=['red', 'gray', 'green'])
    plt.title("Sentiment Probabilities")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    for i, v in enumerate(probs):
        plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig("sentiment_probs.png")
    plt.close()

    return f"Prediction: {labels[class_id]} ({confidence*100:.2f}%)", "sentiment_probs.png"

# 🎛️ Gradio Interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type something like 'U R SUPER 🤩'..."),
    outputs=["text", "image"],
    title="Contextual 3-Class Sentiment Classifier",
    description="Enter casual, emoji-rich text (SMS, chat, tweets). Model uses Twitter-RoBERTa for 3-class sentiment classification with emoji handling. Graph shows class confidence scores."
)

iface.launch()
