
import os
import re
import time
import requests
import openai
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import praw
import base64
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
# Ensure folders are created in the same directory as this script
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)

import google.generativeai as genai
import textwrap
import unicodedata

# ========== SETUP ==========
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Setup OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# Setup Reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Create folders
Path("images").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ========== STEP 1: SCRAPE POSTS ==========
def scrape_posts(limit=10):
    posts = []
    subreddit = reddit.subreddit("whatsthissnake")
    submissions = list(subreddit.top(limit=limit, time_filter="all"))
    if not submissions:
        print("No posts from .top(), trying .hot() instead.")
        submissions = list(subreddit.hot(limit=limit))

    for submission in submissions:
        print(f"Checking post: {submission.title} -> {submission.url}")
        if submission.url.lower().endswith((".jpg", ".jpeg", ".png")):
            submission.comments.replace_more(limit=0)
            top_comment = ""
            reliable_comment = ""
            for comment in submission.comments:
                if hasattr(comment, 'body'):
                    flair = getattr(comment, 'author_flair_richtext', None)
                    if isinstance(flair, list):
                        for f in flair:
                            if isinstance(f, dict) and 't' in f and 'reliable responder' in f['t'].lower():
                                reliable_comment = comment.body
                                break
                    if not top_comment:
                        top_comment = comment.body

            posts.append({
                "id": submission.id,
                "title": submission.title,
                "url": submission.url,
                "top_comment": top_comment,
                "reliable_comment": reliable_comment
            })

    print(f"Total image posts collected: {len(posts)}")
    return posts

# ========== STEP 2: DOWNLOAD IMAGES ==========
def download_images(posts):
    valid_posts = []
    for post in tqdm(posts):
        try:
            response = requests.get(post["url"])
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_path = f"images/{post['id']}.jpg"
            image.save(image_path)
            post["image_path"] = image_path
            valid_posts.append(post)
        except Exception as e:
            post["error"] = str(e)
            print(f"Failed to download {post['url']} - {e}")
    return valid_posts

# ========== STEP 3: CLASSIFY IMAGE ==========
def classify_image(post):
    try:
        with open(post["image_path"], "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a herpetologist. Identify the snake in the image and reply in this format:\n\nCommon Name: [name if known]\nScientific Name: [scientific name if known]\n\nDo not include anything else."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image Title: '{post['title']}'"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ]
        )
        post["gpt_label"] = response.choices[0].message.content.strip()

    except Exception as e:
        post["gpt_label"] = f"ERROR: {e}"
    return post

# ========== STEP 4: GEMINI GROUND TRUTH EXTRACTION ==========
def extract_ground_truth(post):
    comment_text = post.get("reliable_comment") or post.get("top_comment", "")
    if not comment_text.strip():
        post["ground_truth"] = "UNKNOWN"
        return post
    try:
        prompt = f"Extract the snake name (common or scientific) from this comment: {comment_text}\nOnly return the name, no extra words."
        response = model_gemini.generate_content(prompt)
        result = response.text.strip().split("\n")[0]
        post["ground_truth"] = result if result else "UNKNOWN"
    except Exception as e:
        post["ground_truth"] = f"ERROR: {e}"
    return post

# ========== STEP 5: GENERATE PDF DOCUMENT ==========
def safe_text(text, width=100):
    if not text:
        return ""
    text = (
        unicodedata.normalize("NFKD", str(text))
        .encode("latin-1", "ignore")
        .decode("latin-1")
    )
    text = text.replace('\u200b', ' ').replace('\u00a0', ' ').replace('\u202e', '')
    text = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in text)
    text = re.sub(r'(\S{' + str(width) + r',})', lambda m: '\n'.join(textwrap.wrap(m.group(1), width)), text)
    return text

def generate_pdf(data_list, output_path="output.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", size=12)

    for idx, item in enumerate(data_list):
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 16)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 10, f"Snake Report #{idx+1}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

        if item.get("image_path"):
            try:
                pdf.image(item["image_path"], w=100)
                pdf.ln(5)
            except RuntimeError as e:
                pdf.set_font("helvetica", size=10)
                pdf.set_text_color(255, 0, 0)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(w=pdf.w - 2 * pdf.l_margin, h=10, txt=f"[Image Error: {e}]")
                pdf.set_text_color(0, 0, 0)

        for label, key in [
            ("GPT Output", "gpt_label"),
            ("Reliable Responder", "reliable_comment"),
            ("Top Comment", "top_comment"),
            ("Ground Truth", "ground_truth")
        ]:
            content = item.get(key, "[No data]")
            pdf.set_font("helvetica", 'B', 12)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w=pdf.w - 2 * pdf.l_margin, h=10, txt=f"{label}:")
            pdf.set_font("helvetica", size=12)
            pdf.set_x(pdf.l_margin)
            try:
                pdf.multi_cell(w=pdf.w - 2 * pdf.l_margin, h=10, txt=safe_text(content))
            except Exception as e:
                print("\n\n\U0001f6a8 Offending content below:")
                print("Field label:", label)
                print("Raw content:", repr(content))
                for i, c in enumerate(content):
                    print(f"{i}: {repr(c)} (ord={ord(c)})")
                raise
            pdf.ln(3)

        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.5)
        y = pdf.get_y()
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(5)

    pdf.output(output_path)

# ========== STEP 6: EVALUATE ==========
def evaluate(posts):
    df = pd.DataFrame(posts)
    df_filtered = df[df["ground_truth"] != "UNKNOWN"]

    def gemini_check(gt, pred):
        prompt = (
            f"""Ground truth: {gt}
Model prediction: {pred}
Is this a correct match? Just reply 'Yes' or 'No'."""
        )
        try:
            response = model_gemini.generate_content(prompt)
            answer = response.text.strip().lower()
            time.sleep(4)
            return answer.startswith("yes")
        except Exception as e:
            print(f"Gemini error during evaluation: {e}")
            time.sleep(10)
            return False

    df_filtered["correct"] = df_filtered.apply(
        lambda x: gemini_check(x["ground_truth"], x["gpt_label"]), axis=1
    )

    tp = df_filtered["correct"].sum()
    fp = (~df_filtered["correct"]).sum()
    fn = df[df["ground_truth"] == "UNKNOWN"].shape[0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / df_filtered.shape[0] if df_filtered.shape[0] > 0 else 0

    # Save detailed output
    df["correct"] = df["ground_truth"].ne("UNKNOWN") & df.apply(lambda x: gemini_check(x["ground_truth"], x["gpt_label"]) if x["ground_truth"] != "UNKNOWN" else False, axis=1)
    df.to_csv("results/classification_results.csv", index=False)

    return precision, recall, accuracy







# import os
# import re
# import time
# import requests
# import openai
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# from io import BytesIO
# from tqdm import tqdm
# import praw
# import base64
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# from fpdf.enums import XPos, YPos
# import google.generativeai as genai

# # Ensure folders are created in the same directory as this script
# BASE_DIR = Path(__file__).parent.resolve()
# os.chdir(BASE_DIR)

# # ========== SETUP ==========
# REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
# REDDIT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
# REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# # Setup OpenAI
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

# # Setup Gemini
# genai.configure(api_key=GEMINI_API_KEY)
# # Using gemini-2.0-flash for better free-tier limits, but still subject to RPM
# model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# # Setup Reddit
# reddit = praw.Reddit(
#     client_id=REDDIT_CLIENT_ID,
#     client_secret=REDDIT_SECRET,
#     user_agent=REDDIT_USER_AGENT
# )
# # Create folders
# Path("images").mkdir(exist_ok=True)
# Path("results").mkdir(exist_ok=True)

# # ========== STEP 1: SCRAPE POSTS ==========
# def scrape_posts(limit=10):
#     posts = []
#     subreddit = reddit.subreddit("whatsthissnake")
#     submissions = list(subreddit.top(limit=limit, time_filter="all"))
#     if not submissions:
#         print("No posts from .top(), trying .hot() instead.")
#         submissions = list(subreddit.hot(limit=limit))

#     for submission in submissions:
#         print(f"Checking post: {submission.title} -> {submission.url}")
#         if submission.url.lower().endswith((".jpg", ".jpeg", ".png")):
#             submission.comments.replace_more(limit=0)
#             top_comment = ""
#             reliable_comment = ""
#             for comment in submission.comments:
#                 if hasattr(comment, 'body'):
#                     flair = getattr(comment, 'author_flair_richtext', None)
#                     if isinstance(flair, list):
#                         for f in flair:
#                             if isinstance(f, dict) and 't' in f and 'reliable responder' in f['t'].lower():
#                                 reliable_comment = comment.body
#                                 break
#                     if not top_comment:
#                         top_comment = comment.body

#             posts.append({
#                 "id": submission.id,
#                 "title": submission.title,
#                 "url": submission.url,
#                 "top_comment": top_comment,
#                 "reliable_comment": reliable_comment
#             })

#     print(f"Total image posts collected: {len(posts)}")
#     return posts

# # ========== STEP 2: DOWNLOAD IMAGES ==========
# def download_images(posts):
#     valid_posts = []
#     for post in tqdm(posts):
#         try:
#             response = requests.get(post["url"])
#             image = Image.open(BytesIO(response.content)).convert("RGB")
#             image_path = f"images/{post['id']}.jpg"
#             image.save(image_path)
#             post["image_path"] = image_path
#             valid_posts.append(post)
#         except Exception as e:
#             post["error"] = str(e)
#             print(f"Failed to download {post['url']} - {e}")
#     return valid_posts

# # ========== STEP 3: CLASSIFY IMAGE ==========
# def classify_image(post):
#     try:
#         with open(post["image_path"], "rb") as img_file:
#             b64_image = base64.b64encode(img_file.read()).decode("utf-8")

#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a herpetologist. Identify the snake in the image and reply in this format:\n\nCommon Name: [name if known]\nScientific Name: [scientific name if known]\n\nDo not include anything else."},
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": f"Image Title: '{post['title']}'"},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
#                     ]
#                 }
#             ]
#         )
#         post["gpt_label"] = response.choices[0].message.content.strip()

#     except Exception as e:
#         post["gpt_label"] = f"ERROR: {e}"
#     return post

# # ========== STEP 4: GEMINI GROUND TRUTH EXTRACTION ==========
# def extract_ground_truth(post):
#     comment_text = post.get("reliable_comment") or post.get("top_comment", "")
#     if not comment_text.strip():
#         post["ground_truth"] = "UNKNOWN"
#         return post
#     try:
#         prompt = f"Extract the snake name (common or scientific) from this comment: {comment_text}\nOnly return the name, no extra words."
#         response = model_gemini.generate_content(prompt)
#         result = response.text.strip().split("\n")[0]
#         post["ground_truth"] = result if result else "UNKNOWN"
#     except Exception as e:
#         post["ground_truth"] = f"ERROR: {e}"
#     return post

# # ========== STEP 5: GENERATE PDF DOCUMENT ==========
# import textwrap
# import unicodedata

# def safe_text(text, width=100):
#     if not text:
#         return ""
#     text = (
#         unicodedata.normalize("NFKD", str(text))
#         .encode("latin-1", "ignore")
#         .decode("latin-1")
#     )
#     text = text.replace('\u200b', ' ').replace('\u00a0', ' ').replace('\u202e', '')
#     text = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in text)
#     # This regex ensures that words longer than `width` are wrapped
#     text = re.sub(r'(\S{' + str(width) + r',})', lambda m: '\n'.join(textwrap.wrap(m.group(1), width)), text)
#     return text

# def generate_pdf(data_list, output_path="output.pdf"):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("helvetica", size=12)

#     for idx, item in enumerate(data_list):
#         pdf.add_page()
#         pdf.set_font("helvetica", 'B', 16)
#         pdf.set_x(pdf.l_margin)
#         pdf.cell(0, 10, f"Snake Report #{idx+1}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
#         pdf.ln(5)

#         if item.get("image_path"):
#             try:
#                 pdf.image(item["image_path"], w=100)
#                 pdf.ln(5)
#             except RuntimeError as e:
#                 pdf.set_font("helvetica", size=10)
#                 pdf.set_text_color(255, 0, 0)
#                 pdf.set_x(pdf.l_margin)
#                 pdf.multi_cell(w=pdf.w - 2 * pdf.l_margin, h=10, txt=f"[Image Error: {e}]")
#                 pdf.set_text_color(0, 0, 0)

#         for label, key in [
#             ("GPT Output", "gpt_label"), # Corrected key here
#             ("Reliable Responder", "reliable_comment"),
#             ("Top Comment", "top_comment"),
#             ("Ground Truth", "ground_truth")
#         ]:
#             content = item.get(key, "[No data]")
#             pdf.set_font("helvetica", 'B', 12)
#             pdf.set_x(pdf.l_margin)
#             pdf.multi_cell(w=pdf.w - 2 * pdf.l_margin, h=10, txt=f"{label}:")
#             pdf.set_font("helvetica", size=12)
#             pdf.set_x(pdf.l_margin)
#             try:
#                 pdf.multi_cell(w=pdf.w - 2 * pdf.l_margin, h=10, txt=safe_text(content))
#             except Exception as e:
#                 print("\n\n\U0001f6a8 Offending content below:")
#                 print("Field label:", label)
#                 print("Raw content:", repr(content))
#                 for i, c in enumerate(content):
#                     print(f"{i}: {repr(c)} (ord={ord(c)})")
#                 raise
#             pdf.ln(3)

#         pdf.set_draw_color(200, 200, 200)
#         pdf.set_line_width(0.5)
#         y = pdf.get_y()
#         pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
#         pdf.ln(5)

#     pdf.output(output_path)

# # ========== STEP 6: EVALUATE ==========
# def evaluate(posts):
#     df = pd.DataFrame(posts)
#     df_filtered = df[df["ground_truth"] != "UNKNOWN"]

#     def gemini_check(gt, pred):
#         prompt = (
#             f"""Ground truth: {gt}
# Model prediction: {pred}
# Is this a correct match? Just reply 'Yes' or 'No'."""
#         )
#         try:
#             response = model_gemini.generate_content(prompt)
#             answer = response.text.strip().lower()
#             # Add a small delay after each Gemini call to stay within rate limits
#             time.sleep(4) # Adjust this value based on your observed rate limit (15 req/min = 4 seconds/req)
#             return answer.startswith("yes")
#         except Exception as e:
#             print(f"Gemini error during evaluation: {e}")
#             # Consider adding a longer sleep or retry logic here for robustness
#             time.sleep(10) # Sleep longer on error
#             return False

#     # Ensure that 'gpt_output' is mapped to 'gpt_label' if it's not already correct
#     # The previous code had "gpt_output" as a key in generate_pdf, but "gpt_label" is used in classify_image.
#     # We should ensure consistency, either by renaming in classify_image or here.
#     # Assuming 'gpt_label' is the actual key in your 'posts' dictionary.
#     df_filtered["correct"] = df_filtered.apply(
#         lambda x: gemini_check(x["ground_truth"], x["gpt_label"]), axis=1
#     )

#     tp = df_filtered["correct"].sum()
#     fp = (~df_filtered["correct"]).sum()
#     fn = df[df["ground_truth"] == "UNKNOWN"].shape[0]
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     accuracy = tp / df_filtered.shape[0] if df_filtered.shape[0] > 0 else 0
#     return precision, recall, accuracy

# # ========== MAIN ==========
# if __name__ == "__main__":
#     posts = scrape_posts(limit=10) # Increased limit to 100 posts
#     posts = download_images(posts)
#     # Using tqdm for both loops for better progress visualization
#     for i, post in tqdm(enumerate(posts), desc="Processing posts (GPT & Gemini GT)"):
#         post = classify_image(post)
#         post = extract_ground_truth(post)
#         # Adding a sleep here to respect Gemini's rate limits for ground truth extraction
#         # 15 requests/minute means 60/15 = 4 seconds between requests
#         time.sleep(4) # This helps with extract_ground_truth calls

#     df = pd.DataFrame(posts)
#     df.to_csv("results/classification_results.csv", index=False)
#     generate_pdf(posts)

#     print("\nStarting evaluation (Gemini calls will be rate-limited)...")
#     precision, recall, accuracy = evaluate(posts)
#     print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nAccuracy: {accuracy:.2f}")