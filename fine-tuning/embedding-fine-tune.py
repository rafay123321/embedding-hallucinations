from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import os
import logging
import random
import numpy as np
import torch
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

output_path = Path.cwd() / "fine-tuned-semantic-model"
output_path.mkdir(exist_ok=True)

data_path = Path.cwd() / "data"

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Parameters
model_name = 'all-MiniLM-L6-v2'  # Base model to fine-tune
num_epochs = 25
train_batch_size = 32
max_seq_length = 128
warmup_ratio = 0.1


def load_data(file_path):
    """Load data from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    return pd.read_csv(file_path)

def main():
    # Load the base model
    logging.info(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Set max sequence length
    model.max_seq_length = max_seq_length
    
    # Load training data
    logging.info("Loading training data")
    train_df = load_data(f"{data_path}/train.csv")
    val_df = load_data(f"{data_path}/val.csv")
    
    # Convert data to InputExamples
    train_examples = [
        InputExample(texts=[row['sentence1'], row['sentence2']], label=row['similarity']) 
        for _, row in train_df.iterrows()
    ]
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    
    # Set up evaluator
    logging.info("Setting up evaluator")
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=val_df['sentence1'].tolist(),
        sentences2=val_df['sentence2'].tolist(),
        scores=val_df['similarity'].tolist()
    )
    
    # Set up the loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Calculate warmup steps
    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_ratio)
    
    # Train the model
    logging.info(f"Beginning training for {num_epochs} epochs")
    logging.info(f"Beginning training for {output_path} epochs")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        evaluation_steps=len(train_dataloader) // 2,  # Evaluate twice per epoch
        save_best_model=True
    )
    
    # Test on the original problematic pairs
    logging.info("Evaluating on test set")
    test_df = load_data(f"{data_path}/test.csv")
    
    # Load the best model
    best_model = SentenceTransformer(str(output_path))
    
    # Encode the sentences
    embeddings1 = best_model.encode(test_df['sentence1'].tolist())
    embeddings2 = best_model.encode(test_df['sentence2'].tolist())
    
    # Calculate cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = []
    for i in range(len(embeddings1)):
        sim = cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
        similarities.append(sim)
    
    # Add to test_df
    test_df['predicted_similarity'] = similarities
    
    # Print results
    logging.info("Test results:")
    for _, row in test_df.iterrows():
        logging.info(f"Pair: '{row['sentence1']}' vs '{row['sentence2']}'")
        logging.info(f"Predicted similarity: {row['predicted_similarity']:.4f}")
        logging.info("-----")
    
    # Calculate average similarity
    avg_similarity = sum(similarities) / len(similarities)
    logging.info(f"Average similarity on test set: {avg_similarity:.4f}")
    
    # Save test results
    test_df.to_csv(f"{output_path}/test_results.csv", index=False)
    logging.info("Test results saved to test_results.csv")
    
    logging.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()