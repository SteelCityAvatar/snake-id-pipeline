

from snekid import scrape_posts, download_images, classify_image, extract_ground_truth, evaluate, generate_pdf
from tqdm import tqdm
import pandas as pd
import time
# ========== MAIN ==========
if __name__ == "__main__":
    posts = scrape_posts(limit=100) # Increased limit to 100 posts
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
    generate_pdf(posts)

    print("\nStarting evaluation (Gemini calls will be rate-limited)...")
    precision, recall, accuracy = evaluate(posts)
    print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nAccuracy: {accuracy:.2f}")