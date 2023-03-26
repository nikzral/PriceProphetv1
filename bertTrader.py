import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import yfinance as yf

# Set the start and end dates
start_date = "2023-02-25"
end_date = "2023-03-24"

# Download the stock data for AAPL from Yahoo Finance
df = yf.download("AAPL", start=start_date, end=end_date, interval="1d")

# Save the stock data to a CSV file
df.to_csv("AAPL_stock_data.csv")



# Set up API credentials and parameters
news_api_key = "a74ab8e0d0b047129bcfab258890aa8f"
company_ticker = "AAPL"
start_date = "2023-02-25"
end_date = "2023-03-24"

# Set up the NewsAPI request
url = ('https://newsapi.org/v2/everything?'
       'q=' + company_ticker + '&'
       'from=' + start_date + '&'
       'to=' + end_date + '&'
       'sortBy=popularity&'
       'apiKey=' + news_api_key)
print(url)
response = requests.get(url)
try:
    articles = response.json()['articles']
except KeyError:
    print("Error: 'articles' key not found in the API response.")
    
  

# Extract news article data and stock price data
articles = response.json()['articles']
df = pd.read_csv("AAPL_stock_data.csv")
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Clean and preprocess text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    text = text.lower().strip()
    text = ' '.join(text.split())
    tokens = tokenizer.encode_plus(text, max_length=256, pad_to_max_length=True, return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']

input_ids = []
attention_masks = []
labels = []

def predict_price_action(article_text, model, tokenizer):
    # Preprocess the input article
    input_id, attention_mask = preprocess_text(article_text)

    # Convert input_id and attention_mask to numpy arrays and reshape them
    input_id = np.array(input_id).reshape(1, -1)
    attention_mask = np.array(attention_mask).reshape(1, -1)

    # Make the prediction using the trained model
    prediction = model.predict([input_id, attention_mask])

    # Return the predicted price action
    return 1 if prediction[0][0] > 0.5 else 0

for article in articles:
    if article['publishedAt'][:10] in df['Date'].values:
        input_id, attention_mask = preprocess_text(article['description'])
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        stock_price = df.loc[df['Date'] == article['publishedAt'][:10]]['Close'].values[0]
        if stock_price > df.loc[df['Date'] == article['publishedAt'][:10]]['Open'].values[0]:
            labels.append(1)
        else:
            labels.append(0)

# Convert data to numpy arrays
input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)
labels = np.array(labels)

# Split data into train, validation, and test sets
train_input_ids = input_ids[:int(0.8*len(input_ids))]
train_attention_masks = attention_masks[:int(0.8*len(attention_masks))]
train_labels = labels[:int(0.8*len(labels))]
val_input_ids = input_ids[int(0.8*len(input_ids)):int(0.9*len(input_ids))]
val_attention_masks = attention_masks[int(0.8*len(attention_masks)):int(0.9*len(attention_masks))]
val_labels = labels[int(0.8*len(labels)):int(0.9*len(labels))]
test_input_ids = input_ids[int(0.9*len(input_ids)):]
test_attention_masks = attention_masks[int(0.9*len(attention_masks)):]
test_labels = labels[int(0.9*len(labels)):]

# Create the deep learning model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
input_ids_layer = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)
attention_mask_layer = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)

bert_output = bert_model([input_ids_layer, attention_mask_layer])[1]
dense_layer = tf.keras.layers.Dense(64, activation='relu')(bert_output)
dropout_layer = tf.keras.layers.Dropout(0.2)(dense_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_layer)

model = tf.keras.models.Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output_layer)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit([train_input_ids, train_attention_masks],
                    train_labels,
                    validation_data=([val_input_ids, val_attention_masks], val_labels),
                    epochs=3,
                    batch_size=8)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([test_input_ids, test_attention_masks], test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# Input article text
article_text = "Apple’s Tim Cook Upbeat in Beijing as China Courts Global CEOs"

# Predict the price action
predicted_price_action = predict_price_action(article_text, model, tokenizer)

# Print the result
print("Predicted price action:", "Up" if predicted_price_action == 1 else "Down")