# scripts/example_data.py

from transformers import BertForSequenceClassification, BertTokenizer
import torch

def predict_emotion(text_list):
    # Load the saved model and tokenizer
    model = BertForSequenceClassification.from_pretrained('models/saved_model')
    tokenizer = BertTokenizer.from_pretrained('models/saved_model')
    model.eval()

    # Tokenize the input texts
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Map predictions to emotion labels
    label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    predicted_emotions = [label_names[pred] for pred in predictions]

    return predicted_emotions

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "I'm so happy to hear from you!",
        "This is the worst day ever.",
        "I can't wait to see you again.",
        "I'm feeling very anxious about the meeting.",
        "You did an amazing job!",
        "I'm utterly disappointed."
    ]

    predictions = predict_emotion(sample_texts)
    for text, emotion in zip(sample_texts, predictions):
        print(f"Text: {text}\nPredicted Emotion: {emotion}\n")
