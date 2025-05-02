import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Model definition
class AuthorshipVerificationModel(nn.Module):
    def __init__(
        self, 
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Siamese network with shared weights
        self.dropout = nn.Dropout(dropout_rate)
        # Simple linear classifier
        self.classifier = nn.Linear(self.config.hidden_size * 3, 2)  # Using difference and concatenation
        # Multi-layer classifier
        """self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )"""

    def encode_text(self, text: str) -> torch.Tensor:
        #Encode a single text using the transformer model
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.transformer.device) for k, v in inputs.items()}
        
        outputs = self.transformer(**inputs)
            
        # Use CLS token representation as text embedding
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(
        self,
        known_texts: List[str],
        unknown_texts: List[str]
    ) -> torch.Tensor:
        batch_size = len(known_texts)
        known_embeddings = []
        unknown_embeddings = []
        
        for i in range(batch_size):
            # Process each sample's known texts (could be multiple)
            sample_known_texts = known_texts[i]
            sample_unknown_texts = unknown_texts[i]
            
            # Get embeddings for each text and average them
            k_embeds = []
            for text in sample_known_texts:
                k_embeds.append(self.encode_text(text))
            k_embed = torch.mean(torch.cat(k_embeds), dim=0, keepdim=True)
            
            u_embeds = []
            for text in sample_unknown_texts:
                u_embeds.append(self.encode_text(text))
            u_embed = torch.mean(torch.cat(u_embeds), dim=0, keepdim=True)
            
            known_embeddings.append(k_embed)
            unknown_embeddings.append(u_embed)
        
        # Stack all batch embeddings
        known_embeddings = torch.cat(known_embeddings, dim=0)
        unknown_embeddings = torch.cat(unknown_embeddings, dim=0)
            
        # Create feature vector using concatenation and absolute difference
        abs_diff = torch.abs(known_embeddings - unknown_embeddings)
        combined = torch.cat((known_embeddings, unknown_embeddings, abs_diff), dim=1)
        combined = self.dropout(combined)
        
        # Classify whether same author or not
        logits = self.classifier(combined)
        return logits

# Dataset handling
class PAN15Dataset(Dataset):
    def __init__(self, data_dir: str, problem_ids: List[str], is_training: bool = True):
        self.data_dir = data_dir
        self.problem_ids = problem_ids
        self.is_training = is_training
        
        # Load labels from truth.txt if training
        self.labels = {}
        if is_training:
            truth_path = os.path.join(data_dir, "truth.txt")
            if os.path.exists(truth_path):
                with open(truth_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            problem_id, label_str = parts
                            # Y = same author (1), N = different author (0)
                            self.labels[problem_id] = 1 if label_str == 'Y' else 0
                print(f"Loaded {len(self.labels)} labels from truth.txt")
        
        self.samples = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        samples = []
        
        for problem_id in self.problem_ids:
            problem_dir = os.path.join(self.data_dir, problem_id)
            
            # Get label from pre-loaded labels
            label = self.labels.get(problem_id) if self.is_training else None
            
            # Load known and unknown texts
            known_texts = []
            unknown_texts = []
            
            for filename in os.listdir(problem_dir):
                filepath = os.path.join(problem_dir, filename)
                if os.path.isfile(filepath):
                    # Handle known text files
                    if filename.startswith("known") and filename.endswith(".txt"):
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            known_texts.append(f.read())
                    # Handle unknown text files
                    elif filename.startswith("unknown") and filename.endswith(".txt"):
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            unknown_texts.append(f.read())
            
            samples.append({
                "problem_id": problem_id,
                "known_texts": known_texts,
                "unknown_texts": unknown_texts,
                "label": label
            })
            
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

# Evaluation metrics
@dataclass
class EvalPrediction:
    predictions: np.ndarray
    label_ids: np.ndarray

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Convert logits to predictions
    preds = np.argmax(logits, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def collate_fn(batch):
    return {
        "problem_ids": [item["problem_id"] for item in batch],
        "known_texts": [item["known_texts"] for item in batch],
        "unknown_texts": [item["unknown_texts"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long) if all(item["label"] is not None for item in batch) else None
    }

# Training function
def train_model(
    model_name = "bert-base-multilingual-cased",
    data_dir = "./pan15-training/pan15-authorship-verification-training-dataset-dutch-2015-04-19",
    output_dir = "./output",
    batch_size = 8,
    learning_rate = 2e-5,
    num_epochs = 3,
    max_length = 512,
    test_size = 0.2,
    random_seed = 42
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Load problem IDs from contents.json
    with open(os.path.join(data_dir, "contents.json"), 'r') as f:
        contents = json.load(f)
        all_problem_ids = contents["problems"]
    
    # Split into train/val
    train_ids, val_ids = train_test_split(
        all_problem_ids, test_size=test_size, random_state=random_seed
    )
    
    print(f"Training on {len(train_ids)} problems, validating on {len(val_ids)} problems")
    
    # Create datasets and dataloaders
    train_dataset = PAN15Dataset(data_dir, train_ids, is_training=True)
    val_dataset = PAN15Dataset(data_dir, val_ids, is_training=True)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AuthorshipVerificationModel(
        model_name=model_name,
        max_length=max_length
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_pbar:
            known_texts = batch["known_texts"]
            unknown_texts = batch["unknown_texts"]
            
            # Skip this batch if no labels
            if batch["labels"] is None:
                continue
                
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(known_texts, unknown_texts)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_description(f"Training (loss={loss.item():.4f})")
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                known_texts = batch["known_texts"]
                unknown_texts = batch["unknown_texts"]
                
                # Skip this batch if no labels
                if batch["labels"] is None:
                    continue
                
                labels = batch["labels"].to(device)
                
                logits = model(known_texts, unknown_texts)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Skip evaluation if no batches had labels
        if not all_logits:
            print("No validation samples with labels, skipping evaluation")
            continue
            
        all_logits = np.vstack(all_logits)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        eval_pred = EvalPrediction(
            predictions=all_logits,
            label_ids=all_labels
        )
        
        metrics = compute_metrics(eval_pred)
        print(f"Validation metrics: {metrics}")
        
        # Save best model
        if metrics.get("f1", 0.0) > best_f1:
            best_f1 = metrics.get("f1", 0.0)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"New best model saved with F1: {best_f1:.4f}")
    
    return model

# prediction function
def predict(
    model_path: str,
    model_name: str,
    known_text: str,
    unknown_text: str,
    max_length: int = 512
):
    """Determine if two texts are from the same author"""
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuthorshipVerificationModel(
        model_name=model_name,
        max_length=max_length
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare inputs
    known_texts = [[known_text]]  # Double brackets to match expected format
    unknown_texts = [[unknown_text]]
    
    # Add calibration step - check identical text case
    are_identical = known_text.strip() == unknown_text.strip()
    
    # Generate prediction
    with torch.no_grad():
        logits = model(known_texts, unknown_texts)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        same_author_prob = float(probs[0, 1])  # Probability of same author
        
        # Override probability for identical texts
        if are_identical:
            same_author_prob = 1.0
    
    # Print result
    print("--------------------------------")
    print(f"Known text: {known_text}")
    print("--------------------------------")
    print(f"Unknown text: {unknown_text}")
    print("--------------------------------")
    print(f"Classification: {'Same author' if same_author_prob > 0.7 else 'Different authors'}")
    print("--------------------------------")
    
    return same_author_prob

# Prediction function for a single pair
def predict_pair(
    model_path,
    model_name,
    known_text,
    unknown_text,
    max_length = 512
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuthorshipVerificationModel(
        model_name=model_name,
        max_length=max_length
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    

    with torch.no_grad():
        known_texts = [[known_text]]
        unknown_texts = [[unknown_text]]
        
        # Get logits from model
        logits = model(known_texts, unknown_texts)
        
        # Convert to probability
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability of same author
        
        # Return the probability value
        return float(probs[0])

# Main function
if __name__ == "__main__":
    # train the model
    # train_model(
    #     model_name="distilbert-base-cased",
    #     data_dir="./pan15-training/pan15-authorship-verification-training-dataset-english-2015-04-19",
    #     output_dir="./output",
    #     batch_size=64,
    #     learning_rate=2e-5,
    #     num_epochs=10,
    #     max_length=512,
    # )
    
    # prediction function
    probability = predict(
        model_path="./output/best_model.pt",
        model_name="distilbert-base-cased",
        known_text="On Trade, I have decided, for purposes of Fairness, that I will charge a RECIPROCAL Tariff meaning, whatever Countries charge the United States of America, we will charge them - No more, no less! For purposes of this United States Policy, we will consider Countries that use the VAT System, which is far more punitive than a Tariff, to be similar to that of a Tariff. Sending merchandise, product, or anything by any other name through another Country, for purposes of unfairly harming America, will not be accepted. In addition, we will make provision for subsidies provided by Countries in order to take Economic advantage of the United States. Likewise, provisions will be made for Nonmonetary Tariffs and Trade Barriers that some Countries charge in order to keep our product out of their domain or, if they do not even let U.S. businesses operate. We are able to accurately determine the cost of these Nonmonetary Trade Barriers. It is fair to all, no other Country can complain and, in some cases, if a Country feels that the United States would be getting too high a Tariff, all they have to do is reduce or terminate their Tariff against us. There are no Tariffs if you manufacture or build your product in the United States. For many years, the U.S. has been treated unfairly by other Countries, both friend and foe. This System will immediately bring Fairness and Prosperity back into the previously complex and unfair System of Trade. America has helped many Countries throughout the years, at great financial cost. It is now time that these Countries remember this, and treat us fairly – A LEVEL PLAYING FIELD FOR AMERICAN WORKERS. I have instructed my Secretary of State, Secretary of Commerce, Secretary of the Treasury, and United States Trade Representative (USTR) to do all work necessary to deliver RECIPROCITY to our System of Trade!",
        unknown_text="Sleepy Joe Biden, THE WORST PRESIDENT IN THE HISTORY OF THE UNITED STATES, has allowed millions and millions of Criminals, many of them murderers, drug dealers, and people released from prisons and mental institutions from all around the world, to enter our Country through it’s very dangerous and ill conceived Open Border. Sorry, but it’s my job to get these killers and thugs out of here. THAT’S WHAT I GOT ELECTED TO DO. MAGA!",
        max_length=512
    )
    
    # test the prediction function
    # probability2 = predict(
    #     model_path="./output/best_model.pt",
    #     model_name="distilbert-base-cased",
    #     known_text="Hello World! Hello World!",
    #     unknown_text="Hello World! Hello World!",
    #     max_length=512
    # )
    
