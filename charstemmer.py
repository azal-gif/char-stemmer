import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support
from piraye import NormalizerBuilder

# Define hyperparameters and special tokens
hidden_size = 1024
embedding_dim = 512
eos_token = "$"
max_word_length = 25
eos_index = 0
number_of_epochs = 20
test_size = 0.1

# Initialize the normalizer with specified configurations
normalizer = NormalizerBuilder().alphabet_ar().digit_ar().punctuation_ar().diacritic_delete().tokenizing().remove_extra_spaces().build()

# List of Arabic characters and special tokens
Arabic_chars = [
    eos_token, "ب", " ", "چ", "أ", "ک", "ه", "ذ", "إ", "ح", "ؤ", "ۀ", "/", "ئ", ".",
    "و", "ز", "ق", "پ", "ى", "ك", "ف", "ژ", "آ", "ء", "خ", "گ", "ّ", "ل", "ش", "ي",
    "س", "\u200c", "ع", "ض", "+", "ط", "ن", "ث", "ی", "ج", "م", "ة", "ظ", "د", "غ",
    "ت", "ر", "ا", "ص", 'ّ', 'ِ', 'ً', 'ٍ', '-', 'ْ', 'ٌ', 'ُ', 'َ', '/', '+',
]

def pad_sequence(seq, max_len, pad_char):
    """
    Pad a sequence with a specified character to a specified maximum length.

    Args:
        seq (List): The sequence to be padded.
        max_len (int): The maximum length of the padded sequence.
        pad_char (Any): The character to be used for padding.

    Returns:
        torch.Tensor: The padded sequence as a tensor of type torch.long.
    """
    padded_seq = seq[:max_len] + [pad_char] * (max_len - len(seq))
    return torch.tensor(padded_seq, dtype=torch.long)

class ArabicStemmerDataset(Dataset):
    def __init__(self, filepath):
        """
        Initializes an instance of the class with data read from a file.

        Args:
            filepath (str): The path to the file to be read.

        Initializes the `data` attribute of the instance with a list of tuples. Each tuple contains two lists. The first list represents the input sequence, which is created by splitting the first word in each line of the file and appending the end-of-sequence token. The second list represents the target sequence, which is created by splitting the second word in each line of the file and appending the end-of-sequence token.

        Raises:
            None

        Returns:
            None
        """
        self.data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = normalizer.normalize(line)
                words = line.strip().split("\t")
                if len(words) == 2 and len(words[0]) > 0 and len(words[1]) > 0:
                    input_seq = [char for char in words[0]] + [eos_token]
                    target_seq = [char for char in words[1]] + [eos_token]
                    self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the input and target sequences at the specified index as padded tensors of indices.

        :param idx: The index of the sequence to retrieve.
        :type idx: int
        :return: A tuple of two tensors, the first representing the input sequence as a padded tensor of indices, and the second representing the target sequence as a padded tensor of indices.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        input_seq, target_seq = self.data[idx]
        input_indices = pad_sequence(
            [Arabic_chars.index(char) for char in input_seq], max_word_length, eos_index
        )
        target_indices = pad_sequence(
            [Arabic_chars.index(char) for char in target_seq], max_word_length, eos_index
        )
        return input_indices, target_indices

class CharStemmer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes a CharStemmer instance with the given input size, hidden size, and output size.

        Args:
            input_size (int): The size of the input to the CharStemmer.
            hidden_size (int): The size of the hidden layer in the CharStemmer.
            output_size (int): The size of the output layer in the CharStemmer.

        Initializes the CharStemmer instance with the following attributes:
            - embedding: An instance of nn.Embedding, which maps input indices to dense vectors.
            - encoder: An instance of nn.LSTM, which applies a bidirectional LSTM to the embedded input.
            - decoder: An instance of nn.LSTM, which applies a LSTM to the output of the bidirectional LSTM.
            - output_layer: An instance of nn.Linear, which maps the output of the LSTM to the output size.
            - eos_index: The index of the end-of-sequence token in the Arabic_chars list.

        Returns:
            None
        """
        super(CharStemmer, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.eos_index = Arabic_chars.index(eos_token)

    def forward(self, input_seq, target_seq, lengths=None):
        """
        Performs a forward pass through the CharStemmer model.

        Args:
            input_seq (torch.Tensor): The input sequence of characters as a tensor.
            target_seq (torch.Tensor): The target sequence of characters as a tensor.
            lengths (torch.Tensor, optional): The lengths of the input and target sequences. Defaults to None.

        Returns:
            torch.Tensor: The predictions made by the model as a tensor.
        """
        embedded_input = self.embedding(input_seq)
        encoded, _ = self.encoder(embedded_input)
        batch_size = input_seq.size(1)
        decoder_input = torch.zeros(
            1, batch_size, encoded.size(2), device=input_seq.device
        )
        hidden = torch.zeros(
            1, batch_size, hidden_size, device=input_seq.device
        )
        cell_state = torch.zeros(
            1, batch_size, hidden_size, device=input_seq.device
        )
        predictions = []
        for t in range(target_seq.size(0)):
            decoder_output, (hidden, cell_state) = self.decoder(
                decoder_input, (hidden, cell_state)
            )
            prediction = self.output_layer(decoder_output.squeeze(0))
            predictions.append(prediction)
            decoder_input = encoded[t].unsqueeze(0)
        predictions = torch.stack(predictions)
        return predictions

def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy of predictions made by a model.

    Args:
        predictions (torch.Tensor): The predictions made by the model. Shape: (batch_size, sequence_length, num_classes).
        targets (torch.Tensor): The true labels. Shape: (batch_size, sequence_length).

    Returns:
        float: The accuracy of the predictions.
    """
    _, predicted_indices = torch.max(predictions, dim=2)
    correct = (predicted_indices == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

def calculate_metrics(predictions, targets):
    """
    Calculate precision, recall, and F1-score metrics for a given set of predictions and targets.

    Args:
        predictions (torch.Tensor): A tensor of shape (batch_size, sequence_length, num_classes) containing the predicted probabilities for each class.
        targets (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the true labels for each sample.

    Returns:
        tuple: A tuple containing the precision, recall, and F1-score metrics.
    """
    _, predicted_indices = torch.max(predictions, dim=2)
    y_true = targets.cpu().numpy().flatten()
    y_pred = predicted_indices.cpu().numpy().flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters and create model instance
input_size = len(Arabic_chars)
output_size = input_size
model = CharStemmer(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Load and split data into training and test sets
# dataset = ArabicStemmerDataset("/home/azal/NoorStemmer_Gold.txt")
dataset = ArabicStemmerDataset("/home/azal/QuranWords2.txt")
test_size = int(test_size * len(dataset))
train_size = len(dataset) - test_size
train_data, test_data = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and test sets
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training loop
for epoch in range(number_of_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    model.train()  # Set the model to training mode
    for i, (input_seq, target_seq) in enumerate(train_loader):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        predictions = model(input_seq, target_seq)
        loss = criterion(
            predictions.view(-1, predictions.size(-1)), target_seq.reshape(-1)
        )
        accuracy = calculate_accuracy(predictions, target_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    print(f"Epoch: [{epoch+1}/{number_of_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    model_save_path = f"char_stemmer_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Evaluate model on test data
model.eval()
test_loss = 0
test_accuracy = 0
all_targets = []
all_predictions = []
with torch.no_grad():
    for input_seq, target_seq in test_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        predictions = model(input_seq, target_seq)
        loss = criterion(
            predictions.view(-1, predictions.size(-1)), target_seq.reshape(-1)
        )
        accuracy = calculate_accuracy(predictions, target_seq)
        test_loss += loss.item()
        test_accuracy += accuracy
        all_targets.append(target_seq)
        all_predictions.append(predictions)
test_loss /= len(test_loader)
test_accuracy /= len(test_loader)
all_targets = torch.cat(all_targets, dim=0)
all_predictions = torch.cat(all_predictions, dim=0)
precision, recall, f1 = calculate_metrics(all_predictions, all_targets)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1-Score: {f1:.4f}")
