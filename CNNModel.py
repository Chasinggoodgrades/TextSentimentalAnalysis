import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical

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
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(len(emotion_labels), activation='softmax')) # Change this line

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Change this line

# Train model
model.fit(data, labels, epochs=6, validation_split=0.2)

# Load test data
test_data = np.loadtxt('test.txt', delimiter=';', dtype=str)
test_texts = test_data[:, 0]  # Texts are the first column
test_labels = test_data[:, 1]  # Labels are the second column

# Tokenize and pad test_texts
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=data.shape[1])  # Use same maxlen as training data

# Convert test_labels to integers and perform one-hot encoding
test_integer_labels = [reverse_emotion_labels[label] for label in test_labels]
test_labels = to_categorical(test_integer_labels, num_classes=len(emotion_labels))

# Evaluate model on test data
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix




from sklearn.metrics import f1_score, precision_score, confusion_matrix
from keras.callbacks import History

# Initialize a History object to store the training metrics
history = History()

# Train model and store its history
model.fit(data, labels, epochs=5, validation_split=0.2, callbacks=[history])

# Predict emotions for the test set
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Print out the text and the predicted emotion label for each test sample
for i in range(len(test_texts)):
    print(f'Text: {test_texts[i]}, Predicted emotion: {emotion_labels[predicted_labels[i]]}')

## Save into a csv file the text and the predicted emotion label as well as the true emotion label
import csv
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Text', 'Predicted Emotion', 'True Emotion'])
    for i in range(len(test_texts)):
        writer.writerow([test_texts[i], emotion_labels[predicted_labels[i]], emotion_labels[test_integer_labels[i]]])

# Calculate F1-Score and Precision
f1 = f1_score(test_integer_labels, predicted_labels, average='weighted')
precision = precision_score(test_integer_labels, predicted_labels, average='weighted')

print(f'F1-Score: {f1}, Precision: {precision}')

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot F1-Score and Precision
plt.figure(figsize=(12, 6))
plt.bar(['F1-Score', 'Precision'], [f1, precision])
plt.title('F1-Score and Precision')
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(test_integer_labels, predicted_labels)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels.values(), yticklabels=emotion_labels.values())
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import classification_report

# Generate classification report
import pandas as pd
from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(test_integer_labels, predicted_labels, target_names=emotion_labels.values())

print(report)