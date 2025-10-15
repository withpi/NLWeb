from datasets import load_dataset, DatasetDict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

taxonomy = {
  "Beverages": [
    "Coffee Brands",
    "Tea Brands",
    "Energy & Functional Drinks",
    "Water & Juice",
    "Beverage Marketplaces & Subscriptions"
  ],
  "Culinary Ingredients & Pantry": [
    "Spices & Seasonings",
    "International & Regional Ingredients",
    "Baking & Grains",
    "Sauces & Condiments",
    "Specialty & Gourmet"
  ],
  "Packaged Foods & Snacks": [
    "Chocolate & Confectionery",
    "Cookies & Baked Treats",
    "Energy & Nutrition Bars",
    "Cereal & Breakfast",
    "Nuts & Seeds"
  ],
  "Meals & Nutrition": [
    "Meal Kits & Delivery",
    "Meal Replacements & Shakes",
    "Special Diets",
    "Health Foods & Natural"
  ],
  "Nutrition & Supplements": [
    "Vitamins & Minerals",
    "Omega & Heart Health",
    "Joint Support",
    "Nootropics & Cognitive",
    "Superfoods & Greens"
  ],
  "Coffee Equipment & Brewing": [
    "Home Coffee Equipment",
    "Commercial Espresso Equipment",
    "Brewing Methods & Accessories",
    "Subscriptions & Curated Coffee"
  ],
  "Ceramics & Pottery": [
    "Japanese Ceramics & Tableware",
    "Pottery Studios & Art",
    "Pottery Supplies & Tools",
    "Kintsugi & Repair",
    "Ikebana Vases & Floral Ceramics"
  ],
  "Japanese Culture & Goods": [
    "Japanese Kitchenware & Home Goods",
    "Japanese Food & Seasonings",
    "Japanese Tea Specialists",
    "Bento & Lunchware",
    "Japanese Crafts & Artisans"
  ],
  "Cooking & Food Media": [
    "Recipe & Technique Platforms",
    "Regional Cuisine Blogs",
    "Food Magazines & Media",
    "Product & Equipment Reviews",
    "Culinary Education"
  ],
  "Travel & Events": [
    "Travel Reviews & Booking",
    "City Guides",
    "Events & Ticketing",
    "Outdoor Trail Platforms",
    "Outdoor Gear Retailers"
  ],
  "Home & Garden": [
    "Home Furnishings Retail",
    "Home & Garden How-To",
    "Indoor & Smart Gardens",
    "Seeds & Gardening Supplies",
    "Sustainable Drinkware & Storage"
  ],
  "Kitchen Tools & Utensils": [
    "Artisan Utensils",
    "Multi-brand Kitchenware Retailers",
    "Drinkware & Bottles",
    "Japanese Knives & Tools"
  ],
  "Apparel & Lifestyle": [
    "Sustainable Footwear & Apparel",
    "Museum & Cultural Stores",
    "Regional Retail Shops"
  ],
  "Media & Entertainment": [
    "Movie/TV Databases",
    "Media Reviews for Families",
    "Podcasts & Audio Networks",
    "News & Journalism",
    "Product Review Platforms"
  ],
  "Crafts & Sewing": [
    "Sewing & Quilting Supplies",
    "Beginner Craft Kits & Workshops",
    "Wholesale & Professional Supplies"
  ],
  "Restaurants & Dining": [
    "Japanese Fine Dining & Omakase"
  ],
  "Real Estate": [
    "Listings & Market Data"
  ],
  "Education & Research": [
    "University Courses",
    "Research Conferences",
    "Professional/Medical Education"
  ]
}

members = {
  "Coffee Brands": [16, 18, 27, 38, 53, 83, 110, 117, 119, 136, 150, 152],
  "Tea Brands": [76, 149, 154, 126, 46],
  "Energy & Functional Drinks": [1, 22],
  "Water & Juice": [29, 93, 92],
  "Beverage Marketplaces & Subscriptions": [42, 56, 105],

  "Spices & Seasonings": [2, 3, 13, 17, 41, 46, 59, 60, 62, 66, 109, 111, 124],
  "International & Regional Ingredients": [14, 15, 43, 90, 91, 116, 134, 123, 94, 6, 96],
  "Baking & Grains": [33, 55, 112, 133, 135, 39],
  "Sauces & Condiments": [12, 26, 40, 64, 70, 75, 89, 153, 155, 157, 107, 4],
  "Specialty & Gourmet": [107, 6, 96, 123, 4, 39],

  "Chocolate & Confectionery": [118, 122],
  "Cookies & Baked Treats": [36],
  "Energy & Nutrition Bars": [77],
  "Cereal & Breakfast": [0],
  "Nuts & Seeds": [106],

  "Meal Kits & Delivery": [5, 156],
  "Meal Replacements & Shakes": [120, 65],
  "Special Diets": [0, 79, 94],
  "Health Foods & Natural": [108, 113, 63],

  "Vitamins & Minerals": [73, 74, 114, 69],
  "Omega & Heart Health": [19],
  "Joint Support": [57, 137],
  "Nootropics & Cognitive": [78],
  "Superfoods & Greens": [68, 100, 65],

  "Home Coffee Equipment": [54, 80, 148],
  "Commercial Espresso Equipment": [82, 115],
  "Brewing Methods & Accessories": [61, 148],
  "Subscriptions & Curated Coffee": [105],

  "Japanese Ceramics & Tableware": [7, 25, 30, 45, 52, 99, 125, 142, 67],
  "Pottery Studios & Art": [24, 30, 51, 99],
  "Pottery Supplies & Tools": [11, 85, 101, 139, 132],
  "Kintsugi & Repair": [23],
  "Ikebana Vases & Floral Ceramics": [32, 52],

  "Japanese Kitchenware & Home Goods": [142, 143],
  "Japanese Food & Seasonings": [6, 96],
  "Japanese Tea Specialists": [126, 25, 67],
  "Bento & Lunchware": [84],
  "Japanese Crafts & Artisans": [7, 45, 125, 140, 23, 30, 25],

  "Recipe & Technique Platforms": [129, 145, 20, 48, 146],
  "Regional Cuisine Blogs": [20, 48, 103, 146, 104],
  "Food Magazines & Media": [147, 145],
  "Product & Equipment Reviews": [129, 98],
  "Culinary Education": [104, 37],

  "Travel Reviews & Booking": [21],
  "City Guides": [34],
  "Events & Ticketing": [97],
  "Outdoor Trail Platforms": [86],
  "Outdoor Gear Retailers": [47],

  "Home Furnishings Retail": [35],
  "Home & Garden How-To": [131],
  "Indoor & Smart Gardens": [151],
  "Seeds & Gardening Supplies": [95],
  "Sustainable Drinkware & Storage": [28],

  "Artisan Utensils": [138],
  "Multi-brand Kitchenware Retailers": [141, 35, 37, 81],
  "Drinkware & Bottles": [28],
  "Japanese Knives & Tools": [143],

  "Sustainable Footwear & Apparel": [49],
  "Museum & Cultural Stores": [127],
  "Regional Retail Shops": [121],

  "Movie/TV Databases": [9, 10, 72],
  "Media Reviews for Families": [71],
  "Podcasts & Audio Networks": [58],
  "News & Journalism": [130],
  "Product Review Platforms": [98],

  "Sewing & Quilting Supplies": [88],
  "Beginner Craft Kits & Workshops": [8],
  "Wholesale & Professional Supplies": [128],

  "Japanese Fine Dining & Omakase": [31, 44],

  "Listings & Market Data": [50],

  "University Courses": [102],
  "Research Conferences": [144],
  "Professional/Medical Education": [87]
}

# 1) Build helpers: inner → top mapping
inner_to_top: Dict[str, str] = {}
for top, inners in taxonomy.items():
    for inner in inners:
        inner_to_top[inner] = top

# 2) Build index → inner categories map (indices refer to rows of non-shopping dataset)
idx_to_inners: Dict[int, Set[str]] = defaultdict(set)
for inner, idxs in members.items():
    for i in idxs:
        idx_to_inners[i].add(inner)

# 3) Load the non-shopping dataset to map URL ↔ row index
nonshop = load_dataset("withpi/nlweb_nonshoppingsites_contents_and_categories", split="train")
url_list: List[str] = nonshop["url"]
url_to_index: Dict[str, int] = {u: i for i, u in enumerate(url_list)}

# 4) Precompute URL → (pi_cat_1, pi_cat_2)
#    pi_cat_2 = list of inner categories
#    pi_cat_1 = list of top-level categories derived from those inners
url_to_cats: Dict[str, Tuple[Optional[List[str]], Optional[List[str]]]] = {}

for i, u in enumerate(url_list):
    inners = sorted(idx_to_inners.get(i, set()))
    if len(inners) == 0:
        url_to_cats[u] = (None, None)
        continue
    tops = sorted({inner_to_top[inner] for inner in inners if inner in inner_to_top})
    url_to_cats[u] = (tops if tops else None, inners if inners else None)

# 5) Load the target corpus config/split
corpus = load_dataset("withpi/nlweb_who_achint_qrel", "corpus", split="train")

# 6) Build new columns aligned with corpus rows
pi_cat_1_col: List[Optional[List[str]]] = []
pi_cat_2_col: List[Optional[List[str]]] = []

missing_urls = 0
for ex in corpus:
    u = ex.get("url")
    if u in url_to_cats:
        tops, inners = url_to_cats[u]
        pi_cat_1_col.append(tops)    # can be list[str] or None
        pi_cat_2_col.append(inners)  # can be list[str] or None
    else:
        # URL in corpus but not found in non-shopping dataset → None, None
        missing_urls += 1
        pi_cat_1_col.append(None)
        pi_cat_2_col.append(None)

print(f"[info] corpus rows: {len(corpus)} | URLs without mapping: {missing_urls}")

# 7) Add columns
corpus = corpus.add_column("pi_cat_1", pi_cat_1_col)
corpus = corpus.add_column("pi_cat_2", pi_cat_2_col)

# 8) Push as a new config name `annotated-corpus` (private)
#    Make sure you are logged in (e.g., `huggingface-cli login`) or set HF_TOKEN in env.
out = DatasetDict({"train": corpus})
out.push_to_hub("withpi/nlweb_who_achint_qrel", config_name="annotated-corpus", private=True)
print("[done] Pushed new config 'annotated-corpus' with columns pi_cat_1 and pi_cat_2.")