import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Define constants
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 6
BATCH_SIZE = 32
EPOCHS = 5



# Define emotion labels
emotion_labels = {
    'joy': 0,
    'sadness': 1,
    'anger': 2,
    'fear': 3,
    'love': 4,
    'surprise': 5
}

# Load data from train.txt
data = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

# Convert emotion labels to integers
data['emotion'] = data['emotion'].map(emotion_labels)

# Split data into texts and labels
train_texts = data['text'].tolist()
train_labels = data['emotion'].tolist()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print(torch.cuda.is_available())
# Create data loaders
train_data = EmotionDataset(train_texts, train_labels)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# # Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# # Define optimizer
#optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#
# Training loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print('-' * 10)

    total_loss = 0
    model.train()

    progress = tqdm(train_loader, desc='Training', dynamic_ncols=True)
    for batch in progress:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device) # Moves batch of input_ids to CPU
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        progress.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    avg_train_loss = total_loss / len(train_loader)
    print(f'Training loss: {avg_train_loss}\n')

print("Training complete!")

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")

# ##Evaluate the model and generate confusion matrix and classification report
# model.eval()
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for batch in train_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         preds = torch.argmax(outputs.logits, dim=1)
#
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# conf_mat = confusion_matrix(all_labels, all_preds)
# sns.heatmap(conf_mat, annot=True)
# print("Confusion Matrix:")
# plt.show()
#
# report = classification_report(all_labels, all_preds)
# print("Classification Report:")
# print(report)






# # Load data from test.txt
# test_data = pd.read_csv('test.txt', sep=';', header=None, names=['text', 'emotion'])
#
# # Convert emotion labels to integers
# test_data['emotion'] = test_data['emotion'].map(emotion_labels)
#
# # Split data into texts and labels
# test_texts = test_data['text'].tolist()
# test_labels = test_data['emotion'].tolist()
#
# # Create a dataset for testing
# test_dataset = EmotionDataset(test_texts, test_labels)
#
# # Create a DataLoader for the test dataset
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
# # Load the trained model
# model.load_state_dict(torch.load('model.pth'))
# model.to(device)
#
# # Evaluate the model and generate confusion matrix and classification report
# model.eval()
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         preds = torch.argmax(outputs.logits, dim=1)
#
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# conf_mat = confusion_matrix(all_labels, all_preds)
#
# emotion_labels = {
#     0: 'joy',
#     1: 'sadness',
#     2: 'anger',
#     3: 'fear',
#     4: 'love',
#     5: 'surprise'
# }
# # Assuming emotion_labels is a dictionary mapping your integer labels to their string representations
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=emotion_labels.values(), yticklabels=emotion_labels.values())
# plt.title('Confusion Matrix')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
#
#
# report = classification_report(all_labels, all_preds, target_names=emotion_labels.values())
# print("Classification Report:")
# print(report)
#
