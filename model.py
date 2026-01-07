import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import random

def get_opus_data(limit=10000):# no longer tatoeba but opus
	print("Loading OPUS Books dataset (English-Spanish)...")
	
	dataset = load_dataset("opus_books", "en-es", split="train")
	
	sentences = []
	print("Filtering and processing sentences...")
	
	for item in dataset:
		# Item structure: {'id': '0', 'translation': {'en': 'Hello', 'es': 'Hola'}}
		english_text = item['translation']['en']
		
		# basic cleaning: remove super short junk
		if len(english_text) > 5: 
			sentences.append(english_text)
			
		if len(sentences) >= limit:
			break
			
	# Shuffle so we don't get alphabetically ordered sentences
	random.shuffle(sentences)
	return sentences


# MODEL DEFINITION
class CharBiLSTM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(CharBiLSTM, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(
			input_size=embedding_dim, 
			hidden_size=hidden_dim, 
			batch_first=True, 
			bidirectional=True
		)
		self.fc = nn.Linear(hidden_dim * 2, 1) # *2 for bidirectional

	def forward(self, x):
		embedded = self.embedding(x)
		lstm_out, _ = self.lstm(embedded)
		logits = self.fc(lstm_out)
		return logits.squeeze(-1)

# Define Vocabulary (a-z + space + padding)
chars = "abcdefghijklmnopqrstuvwxyz .?!'" # Added punctuation patterns
char_to_ix = {ch: i+1 for i, ch in enumerate(chars)}
ix_to_char = {i+1: ch for i, ch in enumerate(chars)}
vocab_size = len(chars) + 1

def prepare_sequence(text):
	"""Converts 'hello' -> Tensor([8, 5, 12, 12, 15])"""
	idxs = [char_to_ix[ch] for ch in text.lower() if ch in char_to_ix]
	return torch.tensor(idxs, dtype=torch.long).unsqueeze(0)

def create_target(text):
	"""
	Input: "my name"
	Output Target: [0, 1, 0, 0, 0, 1] (1 indicates a space follows this char)
	Output InputStr: "myname"
	"""
	text = text.lower()
	clean_input = ""
	labels = []
	
	# If we see a space, the PREVIOUS char gets label 1.
	skip_next = False
	for i, ch in enumerate(text):
		if skip_next:
			skip_next = False
			continue
			
		if ch == " ":
			# Marks previous char as needing a space
			if len(labels) > 0:
				labels[-1] = 1.0
		elif ch in char_to_ix:
			clean_input += ch
			labels.append(0.0)
			
	return torch.tensor(labels, dtype=torch.float).unsqueeze(0), clean_input

if __name__ == "__main__":
	# Configuration of model learning parameters
	EMBEDDING_DIM = 32		# bigger model runs 64
	HIDDEN_DIM = 64 		# bigger model runs 128
	LEARNING_RATE = 0.005 	# bigger model runs 0.001
	EPOCHS = 3 				# bigger model runs 5
	DATA_LIMIT = 5000 		# bigger model runs 50k

	# Load Data
	raw_data = get_opus_data(limit=DATA_LIMIT)
	print(f"Sample data: {raw_data[0:3]}")

	# Initialize Model
	model = CharBiLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
	loss_function = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	# Training Loop
	print("\nStarting training...")
	model.train()
	for epoch in range(EPOCHS):
		total_loss = 0
		for sentence in raw_data:
			model.zero_grad()
			
			# Prepare data
			targets, input_str = create_target(sentence)
			if len(input_str) < 2: continue # Skip single chars
			
			inputs = prepare_sequence(input_str)
			
			# Forward + Backward
			predictions = model(inputs)
			
			# Truncate to match shortest length (safety check)
			min_len = min(predictions.shape[1], targets.shape[1])
			predictions = predictions[:, :min_len]
			targets = targets[:, :min_len]

			loss = loss_function(predictions, targets)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			
		print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(raw_data):.4f}")


	print("Saving model checkpoint...")

	checkpoint = {
		# The Model Weights
		'model_state_dict': model.state_dict(),
		
		# The Vocabulary
		'char_to_ix': char_to_ix,
		'ix_to_char': ix_to_char,
		
		# The Architecture config (for rebuilding the model)
		'config': {
			'vocab_size': vocab_size,
			'embedding_dim': EMBEDDING_DIM,
			'hidden_dim': HIDDEN_DIM
		}
	}

	torch.save(checkpoint, "segmentation_model_smaller.pth")
	print("Saved to 'segmentation_model.pth'")


	#  Interactive CLI Testing
	print("\nTraining Complete! Try typing strings without spaces (e.g., 'thisisasentence').")
	print("Type 'exit' to quit.")
	
	model.eval()
	with torch.no_grad():
		while True:
			user_in = input("\nEnter string: ")
			if user_in.lower() == 'exit': break
			
			# Run Inference
			inputs = prepare_sequence(user_in)
			logits = model(inputs)
			probs = torch.sigmoid(logits)[0] # First batch item
			
			# Reconstruct String
			output = ""
			for i, ch in enumerate(user_in):
				if ch.lower() not in char_to_ix: continue
				output += ch
				# If model is confident (>0.5) and not the last char
				if i < len(probs) and probs[i] > 0.5:
					output += " "
			
			print(f"Segmented:  {output}")