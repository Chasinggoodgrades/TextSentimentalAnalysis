import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load data
data = np.loadtxt('train.txt', delimiter=';', dtype=str)
texts = data[:, 0]  # Texts are the first column
labels = data[:, 1]  # Labels are the second column

# Tokenize texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
data = pad_sequences(sequences)

# Define label mapping
emotion_labels = {
    0: 'joy',
    1: 'sadness',
    2: 'anger',
    3: 'fear',
    4: 'love',
    5: 'surprise'
}

# Reverse the emotion_labels dictionary
reverse_emotion_labels = {v: k for k, v in emotion_labels.items()}

# Convert labels to integers using the reversed dictionary
integer_labels = [reverse_emotion_labels[label] for label in labels]

# Perform one-hot encoding
labels = to_categorical(integer_labels, num_classes=len(emotion_labels))

# Define model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=data.shape[1]))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(len(emotion_labels), activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(data, labels, epochs=5, validation_split=0.2)

# Load test data
test_data = np.loadtxt('test.txt', delimiter=';', dtype=str)
test_texts = test_data[:, 0]  # Texts are the first column
test_labels = test_data[:, 1]  # Labels are the second column

# Tokenize and pad test_texts
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=data.shape[1])  # Use the same maxlen as training data

# Convert test_labels to integers and perform one-hot encoding
test_integer_labels = [reverse_emotion_labels[label] for label in test_labels]
test_labels = to_categorical(test_integer_labels, num_classes=len(emotion_labels))

# Evaluate model on test data
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Predict emotions for the test set
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Print out the text and the predicted emotion label for each test sample
for i in range(len(test_texts)):
    print(f'Text: {test_texts[i]}, Predicted emotion: {emotion_labels[predicted_labels[i]]}')

# Generate classification report
report = classification_report(test_integer_labels, predicted_labels, target_names=emotion_labels.values())
print(report)

# Calculate confusion matrix
cm = confusion_matrix(test_integer_labels, predicted_labels)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels.values(), yticklabels=emotion_labels.values())
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
