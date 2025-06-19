from snekid import scrape_posts, download_images, classify_image, extract_ground_truth, evaluate
from tqdm import tqdm
import pandas as pd
import time
import os

# ========== MAIN ==========
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    posts = scrape_posts(limit=100)  # Increased limit to 100 posts
    posts = download_images(posts)
    
    # Using tqdm for both loops for better progress visualization
    for i, post in tqdm(enumerate(posts), desc="Processing posts (GPT & Gemini GT)"):
        post = classify_image(post)
        post = extract_ground_truth(post)
        # Adding a sleep here to respect Gemini's rate limits for ground truth extraction
        # 15 requests/minute means 60/15 = 4 seconds between requests
        time.sleep(4) # This helps with extract_ground_truth calls

    df = pd.DataFrame(posts)
    df.to_csv("results/classification_results.csv", index=False)

    print("\nStarting evaluation (Gemini calls will be rate-limited)...")
    precision, recall, accuracy, f1_score, results = evaluate(posts)
    print("\n=== Evaluation Metrics ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"\nTotal Known Cases: {results['total_known']}")
    print(f"Total Unknown Cases: {results['total_unknown']}")
    print(f"True Positives: {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    