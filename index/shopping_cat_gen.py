import os
import json
import time
from typing import Dict, Any, List
from datasets import load_dataset
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
DATASET = "withpi/nlweb_who_achint_qrel"
CONFIG = "annotated-corpus"
SPLIT = "train"
DOMAIN_FILTER = "myshopify.com"
GROUP_SIZE = 80

MODEL = "gpt-5"  # Use your preferred GPT-5 model name
THINKING_EFFORT = "medium"  # "low" | "medium" | "high"
# TEMPERATURE = 0.5

# Control which groups to process
START_GROUP_INDEX = 6          # inclusive
MAX_GROUPS = None              # e.g., set to 3 to only run 3 groups; None = run all remaining

# Initial taxonomy (seed)
CURRENT_TAXONOMY: Dict[str, List[str]] = {
  "Ceramics & Pottery": [
    "Traditional & Regional",
    "Contemporary & Functional",
    "Studios & Education"
  ],
  "Food & Beverage": [
    "Pantry & Specialty Foods",
    "Coffee, Tea & Beverages",
    "Food Education & Experiences",
    "Baked Goods & Confections",
    "Sauces, Spices & Oils"
  ],
  "Health & Wellness": [
    "Supplements",
    "Sports Nutrition",
    "Medical & Dental Supplies",
    "Pet & Livestock Care",
    "Herbal & Botanical Remedies"
  ],
  "Kitchen & Culinary Tools": [
    "Knives & Cutlery",
    "Cookware, Bakeware & Kitchenware",
    "Outdoor Grilling",
    "Coffee & Espresso Equipment",
    "Prep Tools & Gadgets"
  ],
  "Bags & Leather": [
    "Fashion & Designer",
    "Canvas & Heritage",
    "Leathercraft Supplies",
    "Eco & Reusable",
    "Travel & Utility"
  ],
  "Textiles & Needlecraft": [
    "Fabrics & Quilting",
    "Yarn & Fiber",
    "Embroidery, Sewing & Printing",
    "Workshops & Classes"
  ],
  "Apparel & Accessories": [
    "Footwear",
    "Headwear & Accessories",
    "Fashion & Workwear",
    "Team & Fan Merch",
    "Intimates & Loungewear"
  ],
  "Home & Lifestyle": [
    "Decor & Fragrance",
    "Furniture, Bedding & Outdoor",
    "Appliances & Gadgets",
    "Home Improvement & Hardware",
    "Tableware & Drinkware"
  ],
  "Beauty & Personal Care": [
    "Skincare",
    "Men's Grooming",
    "Hair, Fragrance & Body",
    "Baby & Family Care",
    "Cosmetics & Tools"
  ],
  "General & Misc": [
    "Variety Stores",
    "Demos & Placeholders",
    "Niche Accessories",
    "Memberships & Tickets"
  ],
  "Sports & Outdoor": [
    "Team Sports Gear",
    "Action Sports (Skate, Cycle, Run)",
    "Fishing & Water",
    "Camps & Experiences"
  ],
  "Books & Media": [
    "Books & Ebooks",
    "Music & Memorabilia",
    "Reports & Directories",
    "Courses & Tutorials",
    "Stationery & Printables"
  ],
  "Business & Industrial": [
    "Packaging & Shipping",
    "Workshop & Automotive",
    "Materials & Samples",
    "Janitorial, Paper & PPE",
    "Printing & Coding Supplies"
  ],
  "Toys, Hobbies & Collectibles": [
    "Action Figures & Models",
    "Trading Cards & TCG",
    "Puzzles & Brain Teasers",
    "DIY Toys & Slimes",
    "Pop Culture Collectibles"
  ],
  "Jewelry & Watches": [
    "Fine Jewelry",
    "Fashion Jewelry",
    "Engagement & Bridal",
    "Loose Gems & Stones",
    "Accessories & Repairs"
  ]
}

# System prompt wrapper. You can tweak tone/criteria here.
SYSTEM_TEMPLATE = """You are an expert data scientist creating an information taxonomy of the web. You will be given an existing taxonomy as a starting point, and you will update it to incorporate new information.

Specifically, you will be given a list of boutique shopify store websites and brief descriptions (first 512 characters only) of them. You will update the provided taxonomy to incorporate these new websites nicely.

There should be two levels to the taxonomy: a top-level of supercategories, and a finer-grained set of categories within each supercategory. The taxonomy given to you will have 18 or fewer supercategories, and 5 or fewer finer-grained categories within each supercategory. The taxonomy you come up with _must_ incorporate all of the categories given to you, but you are allowed to combine them, reword them, etc., in order to merge the new sites into the taxonomy. You can add at MOST 2 extra supercategories; i.e., you can return a taxonomy with at most 20 supercategories. You are limited to 5 finer-grained categories per supercategory.

Items could fall into multiple supercategories, potentially. Think of it as a probability distribution: an item could be 80 percent in one, 20 percent in another.

The goal is to come up with the smallest set of categories that nicely covers the range of all items provided to you (and still has a place for any items you could imagine fitting into the original taxonomy).

Output your taxonomy as a python dictionary where the keys are the category names, and values are lists of the subcategory names. Category names should be at most a few words -- without descriptions.

 The response MUST be a valid JSON object of the updated taxonomy.

Existing taxonomy JSON:
{taxonomy_json}
"""

# User message wrapper. Keep payload tight & deterministic.
USER_TEMPLATE = """{items_json}"""

# -----------------------------
# Helpers
# -----------------------------
def make_batches(ds, size):
    for i in range(0, len(ds), size):
        yield i // size, ds.select(range(i, min(i + size, len(ds))))

def compact_item(ex: dict) -> dict:
    # ex['corpus_text'] is JSON; parse and return minimal fields
    raw = ex.get("corpus_text", "")
    try:
        obj = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        obj = {}
    name = (obj.get("name") or "").strip()
    url = (obj.get("url") or ex.get("url") or "").strip()
    desc = (obj.get("description") or "").strip()
    return {
        "name": name,
        "url": url,
        "description_512": desc[:512]
    }

def force_json(text: str) -> Any:
    """
    Robustly coerce the model output to JSON.
    We request `response_format={"type":"json_object"}`, but keep a fallback.
    """
    try:
        return json.loads(text)
    except Exception:
        # Fallback: try to extract the first {...} block
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))

# -----------------------------
# Main
# -----------------------------
def main():
    client = OpenAI()  # reads OPENAI_API_KEY

    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    ds = ds.filter(lambda ex: isinstance(ex.get("url"), str) and DOMAIN_FILTER in ex["url"])

    # Build batches of size 80
    batches = list(make_batches(ds, GROUP_SIZE))
    if START_GROUP_INDEX >= len(batches):
        raise IndexError(f"START_GROUP_INDEX {START_GROUP_INDEX} out of range; only {len(batches)} groups.")

    # Apply max-groups limit
    end_ix = len(batches) if MAX_GROUPS is None else min(len(batches), START_GROUP_INDEX + MAX_GROUPS)
    current_tax = CURRENT_TAXONOMY

    for batch_ix in range(START_GROUP_INDEX, end_ix):
        _, subset = batches[batch_ix]
        items = [compact_item(ex) for ex in subset]

        system_msg = SYSTEM_TEMPLATE.format(taxonomy_json=json.dumps(current_tax, ensure_ascii=False))
        user_msg = USER_TEMPLATE.format(
            batch_index=batch_ix,
            items_json=json.dumps(items, ensure_ascii=False)
        )

        # Call GPT-5 with JSON-only response
        resp = client.responses.create(
            model=MODEL,
            reasoning={"effort": THINKING_EFFORT},
            #temperature=TEMPERATURE,
            #response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        # Get text; Responses API returns a structured object.
        # Use `output_text` convenience if present; otherwise assemble from content.
        try:
            out_text = resp.output_text
        except AttributeError:
            # Fallback assemble
            parts = []
            for item in getattr(resp, "output", []):
                if getattr(item, "type", "") == "message" and getattr(item, "content", None):
                    for blk in item.content:
                        if blk.get("type") == "output_text":
                            parts.append(blk.get("text", ""))
            out_text = "".join(parts).strip()

        # Parse JSON and update taxonomy
        new_tax = force_json(out_text)
        if not isinstance(new_tax, dict):
            raise ValueError("Model returned JSON that is not an object at top level.")

        current_tax = new_tax

        # Print the new taxonomy between calls
        print(f"\n=== Updated taxonomy after batch {batch_ix} ===")
        print(json.dumps(current_tax, ensure_ascii=False, indent=2))

        # Gentle pacing in case of rate limits
        time.sleep(3)

    # Optionally: write final taxonomy to a file
    with open("final_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(current_tax, f, ensure_ascii=False, indent=2)
    print("\nSaved final taxonomy to final_taxonomy.json")

if __name__ == "__main__":
    main()