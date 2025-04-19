from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_model(model_name, test_data_path, output_path=None):
    """
    Evaluate embedding model on semantic contradiction test data
    
    Args:
        model_name: Name or path of the sentence-transformer model
        test_data_path: Path to the test CSV file
        output_path: Optional path to save results
    """
    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Encode sentences
    print("Encoding sentences...")
    embeddings1 = model.encode(test_df['sentence1'].tolist())
    embeddings2 = model.encode(test_df['sentence2'].tolist())
    
    # Calculate cosine similarities
    similarities = []
    for i in range(len(embeddings1)):
        sim = cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
        similarities.append(sim)
    
    # Add to dataframe
    test_df['predicted_similarity'] = similarities
    
    # Calculate error metrics
    test_df['error'] = abs(test_df['similarity'] - test_df['predicted_similarity'])
    
    # Print results
    print("\nResults Summary:")
    print(f"Mean Absolute Error: {test_df['error'].mean():.4f}")
    print(f"Mean Similarity for Contradictions (target=0.0): {test_df[test_df['similarity'] == 0.0]['predicted_similarity'].mean():.4f}")
    print(f"Mean Similarity for Paraphrases (target=1.0): {test_df[test_df['similarity'] == 1.0]['predicted_similarity'].mean():.4f}")
    
    # List pairs with highest similarity that should be low
    print("\nTop contradictions still scored as similar:")
    contradictions = test_df[test_df['similarity'] == 0.0].sort_values('predicted_similarity', ascending=False)
    for _, row in contradictions.head(5).iterrows():
        print(f"  Similarity: {row['predicted_similarity']:.4f} - '{row['sentence1']}' vs '{row['sentence2']}'")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(
        test_df[test_df['similarity'] == 0.0]['predicted_similarity'],
        test_df[test_df['similarity'] == 0.0]['error'],
        color='red', alpha=0.7, label='Contradictions'
    )
    plt.scatter(
        test_df[test_df['similarity'] == 1.0]['predicted_similarity'],
        test_df[test_df['similarity'] == 1.0]['error'],
        color='blue', alpha=0.7, label='Paraphrases'
    )
    plt.axvline(x=0.5, color='gray', linestyle='--')
    plt.xlabel('Predicted Similarity')
    plt.ylabel('Error')
    plt.title(f'Embedding Model Evaluation: {model_name}')
    plt.legend()
    
    # Save results if requested
    if output_path:
        test_df.to_csv(output_path, index=False)
        plt.savefig(f"{output_path.split('.')[0]}_plot.png")
        print(f"Results saved to {output_path}")
    
    plt.show()
    
    return test_df

if __name__ == "__main__":

    model_path = str(Path.cwd() / "fine-tuned-semantic-model")
    data_path = str(Path.cwd() / "data")
    results_path = Path.cwd() / "results"
    results_path.mkdir(exist_ok=True)

    evaluate_model("sentence-transformers/all-MiniLM-L6-v2", f"{data_path}/test.csv", f"{str(results_path)}/baseline_results.csv")
    evaluate_model(model_path, f"{data_path}/test.csv", f"{str(results_path)}/fine_tuned_results.csv")