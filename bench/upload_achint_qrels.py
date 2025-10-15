from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from urllib.parse import urlparse, urlunparse
import json
import re
import os
from typing import Dict, Any, List, Tuple

REPO_ID = "withpi/nlweb_who_achint_qrel"
TARGET_CONFIG = "achint-qrels"
TARGET_SPLIT = "train"

# ---- 1) Provide your ratings dict here (or load from a file) ----
# Example: RATINGS_JSON = { "0": {"https://example.com": 1.0}, "1": {...}, ... }
# If you prefer a file, set RATINGS_JSON_PATH and leave RATINGS_JSON=None.
RATINGS_JSON: Dict[str, Dict[str, float]] = {
  # 0: I am interested in chai with interesting spices
  "0": {
    "https://https://strand-tea-company.myshopify.com": 1.0,
    "https://marketspice_com": 1.0,
    "https://https://black-majestea-llc.myshopify.com": 1.0,
    "https://strand-tea-company.myshopify.com": 1.0
  },
  # 1: equipment for making coffee, especially light roasts from South America
  "1": {
    "https://https://mule-coffee.myshopify.com": 1.0,
    "https://https://espresso-doctor.myshopify.com": 1.0,
    "https://onyxcoffeelab.com": 1.0,
    "https://https://hot-cup-more.myshopify.com": 1.0,
    "https://alternativebrewing.com.au": 1.0,
    "https://v60.coffee": 0.5,
    "https://seattlecoffeegear.com": 1.0,
    "https://majestycoffee.com": 1.0,
    "https://https://sorrentinacoffee.myshopify.com": 1.0,
    "https://vervecoffee.com": 1.0,
    "https://clivecoffee.com": 1.0
  },
  # 2: equipment for making tandoori roti at home. I am especially interested in the ones that don't require the dough to be fermented for a long time
  "2": {
    "https://https://wood-stone-corporation.myshopify.com": 0.5,
    "https://tandoorstore.myshopify.com": 1.0,
    "https://shopmajestic-cookware.myshopify.com": 1.0,
    "https://wood-stone-corporation.myshopify.com": 1.0,
    "https://craftenka-store.myshopify.com": 0.5,
    "https://burnproject.myshopify.com": 1.0,
    "https://https://cattailswoodwork.myshopify.com": 0.5
  },
  # 3: gifts that would be appropriate for toddlers
  "3": {
    "https://https://deltaplayground.myshopify.com": 1.0,
    "https://https://shopsimplybaby.myshopify.com": 1.0,
    "https://https://deltachildrenstore.myshopify.com": 1.0,
    "https://https://ebeanstalk.myshopify.com": 1.0,
    "https://https://tinyacethreads.myshopify.com": 1.0,
    "https://https://zootampa.myshopify.com": 1.0,
    "https://https://peaberry-kids.myshopify.com": 1.0,
    "https://https://bhxqrg-a0.myshopify.com": 1.0,
    "https://https://imaginuity.myshopify.com": 1.0,
    "https://https://south-coast-baby-co.myshopify.com": 1.0,
    "https://https://littlesproutboutique.myshopify.com": 1.0,
    "https://https://littleonesgs.myshopify.com": 1.0,
    "https://https://puzzle-maniac.myshopify.com": 1.0,
    "https://https://solenneworld.myshopify.com": 1.0,
    "https://https://little-bins-for-little-hands.myshopify.com": 1.0,
    "https://https://oak-and-willow-designs.myshopify.com": 1.0,
    "https://https://wearemakeruk.myshopify.com": 1.0,
    "https://https://doggyland-kids.myshopify.com": 1.0,
    "https://https://customplush.myshopify.com": 1.0,
    "https://https://ceaco.myshopify.com": 1.0,
    "https://https://52toysglobalshop.myshopify.com": 1.0,
    "https://https://axisarbor.myshopify.com": 1.0,
    "https://https://ap-gaming7.myshopify.com": 1.0,
    "https://https://shopbosfiremuseum.myshopify.com": 1.0,
    "https://https://minecraftshopmkds-com.myshopify.com": 1.0
  },
  # 4: I am interested in making gluten free bread. What kind of special equipment do I need?
  "4": {
    "https://https://novaspantry.myshopify.com": 1.0,
    "https://freeflowerco.myshopify.com": 0.5,
    "https://the-bakers-bin.myshopify.com": 1.0,
    "https://https://wood-stone-corporation.myshopify.com": 0.5,
    "https://wood-stone-corporation.myshopify.com": 0.5
  },
  # 5: I am interested in making Kyoto style pottery. Where can I find the kind of blue glaze they use?
  "5": {
    "https://seattle-pottery-supply.myshopify.com": 1.0,
    "https://penguinpottery.myshopify.com": 1.0,
    "https://roadrunnerceramics-2.myshopify.com": 1.0,
    "https://https://ceramic-supply-chicago.myshopify.com": 1.0,
    "https://https://ceramic-supply-inc.myshopify.com": 1.0,
    "https://thepotterscenter.com": 1.0,
    "https://ceramic-supply-inc.myshopify.com": 1.0,
    "https://https://vogueporcelainproducts.myshopify.com": 1.0
  },
  # 6: I like making jams when there are plenty of fruits and keeping them for the winter. I need nice containers for storing them
  "6": {
    "https://theme425-kitchen-supplies.myshopify.com": 1.0,
    "https://cultured-living.myshopify.com": 1.0
  },
  # 7: interesting tea varieties with spices
  "7": {
    "https://https://especias-del-sol.myshopify.com": 1.0,
    "https://https://strand-tea-company.myshopify.com": 1.0,
    "https://strand-tea-company.myshopify.com": 1.0,
    "https://https://atobritain.myshopify.com": 1.0,
    "https://https://edinburgh-tea-coffee.myshopify.com": 1.0,
    "https://https://tienxi.myshopify.com": 1.0,
    "https://https://soleyateas.myshopify.com": 1.0,
    "https://https://mountaintea.myshopify.com": 1.0,
    "https://https://4jscoffeeshop.myshopify.com": 1.0,
    "https://harney.com": 1.0,
    "https://https://gnat-and-bee.myshopify.com": 0.5,
    "https://https://blackthornsbotanicals.myshopify.com": 1.0,
    "https://https://kettleworksteashoppe.myshopify.com": 1.0,
    "https://worldspicemerchants.myshopify.com": 1.0,
    "https://https://qa9q0i-5f.myshopify.com": 1.0,
    "https://https://forsythcoffee.myshopify.com": 0.5,
    "https://https://black-majestea-llc.myshopify.com": 1.0,
    "https://https://greatmsteacompany.myshopify.com": 1.0,
    "https://https://spirit-tea.myshopify.com": 1.0,
    "https://https://theteacharmer-com.myshopify.com": 1.0,
    "https://kalustyans.com": 0.5,
    "https://birdandblendtea.com": 1.0,
    "https://https://teaandcoffeworld.myshopify.com": 1.0,
    "https://https://the-missouri-tea-company.myshopify.com": 1.0,
    "https://makkah-market.myshopify.com": 0.5,
    "https://https://stoeckli-organics.myshopify.com": 1.0,
    "https://chinateapavilion.myshopify.com": 1.0,
    "https://https://soul-fire-farm.myshopify.com": 0.5,
    "https://julytea.no": 1.0
  },
  # 8: My friend is into sewing. She is especially interested in making bags that can be used to carry laptops. Suggest gifts for her
  "8": {
    "https://https://cherishing-today.myshopify.com": 1.0,
    "https://leabu-sewing-center.myshopify.com": 1.0,
    "https://https://tamarasjoy.myshopify.com": 0.5,
    "https://craftyangelshop.myshopify.com": 1.0
  },
  # 9: olive oil cake recipes
  "9": {
    "https://mediterranean_dish": 1.0,
    "https://delish": 1.0,
    "https://bon_appetit": 1.0,
    "https://nytimes": 0.5,
    "https://omnivorebooks.myshopify.com": 0.5,
    "https://chefd.com": 0.5,
    "https://seriouseats": 1.0
  },
  # 10: houses for sale in my area under 750k
  "10": {
    "https://zillow": 1.0
  },
  # 11: Smoked salmon cured with dill and brown sugar
  "11": {
    "https://https://brixham-fishmonger.myshopify.com": 1.0,
    "https://https://orcas-food-coop.myshopify.com": 1.0,
    "https://https://finest-at-sea.myshopify.com": 0.5,
    "https://https://bradysoysters.myshopify.com": 1.0,
    "https://pan-chancho-bakery.myshopify.com": 1.0,
    "https://https://svenfish-test-store.myshopify.com": 0.5,
    "https://https://medford-seafood-market.myshopify.com": 1.0,
    "https://https://seafood-express-inc.myshopify.com": 0.5,
    "https://orcas-food-coop.myshopify.com": 0.5
  },
  # 12: Mediterranean herbs like oregano and thyme from Greece
  "12": {
    "https://https://gus-greek-seasoning.myshopify.com": 1.0,
    "https://https://pylianestates.myshopify.com": 0.5,
    "https://https://stoeckli-organics.myshopify.com": 1.0,
    "https://wholespice.myshopify.com": 1.0,
    "https://https://mannamills.myshopify.com": 1.0,
    "https://https://waldoherbs.myshopify.com": 1.0,
    "https://coolrunningsfoods.com": 1.0,
    "https://https://tm-shopify010-spice.myshopify.com": 1.0,
    "https://theme124-spice-shop.myshopify.com": 1.0,
    "https://oldtownspices.com": 1.0,
    "https://https://leightys-farm-market.myshopify.com": 1.0,
    "https://oaktownspiceshop.com": 0.5,
    "https://burlapandbarrel.com": 0.5,
    "https://thespiceway.com": 1.0,
    "https://worldspicemerchants.myshopify.com": 1.0,
    "https://savoryspiceshop.com": 1.0,
    "https://alquimistes.myshopify.com": 1.0,
    "https://atlastradingonline.myshopify.com": 0.5,
    "https://gneissspice.com": 1.0,
    "https://especias-del-sol.myshopify.com": 0.5,
    "https://okiespice-and-trade-co.myshopify.com": 0.5,
    "https://the-spice-guy.myshopify.com": 0.5,
    "https://spiceology.com": 0.5,
    "https://spice-done-right.myshopify.com": 1.0,
    "https://burlap-barrel.myshopify.com": 0.5,
    "https://rumispice.com": 0.5
  },
  # 13: Podcasts about teenage screen usage
  "13": {
    "https://commonsensemedia": 0.5,
    "https://npr_podcasts": 1.0,
    "https://nytimes": 1.0,
    "https://med_podcast": 0.5
  },
  # 14: Movies for teenagers about shopping or food
  "14": {
    "https://imdb": 1.0,
    "https://commonsensemedia": 0.5,
    "https://movie_data": 1.0,
    "https://scifi_movies": 0.5
  },
  # 15: jam filled gluten free cake recipe
  "15": {
    "https://delish": 1.0,
    "https://bon_appetit": 1.0,
    "https://chefd.com": 0.5
  },
  # 16: trails with olive trees in italy
  "16": {
    "https://alltrails": 1.0,
    "https://tripadvisor": 1.0
  },
  # 17: I have trouble sleeping. can you give a list of top headphones that can act as sleeping aid
  "17": {
    "https://wirecutter": 1.0,
    "https://https://os-trnz.myshopify.com": 0.5,
    "https://https://mbt4hs-xg.myshopify.com": 0.5
  },
  # 18: I got some spices from a shop, now I want to make some delicious food. Can you recommend recipes
  "18": {
    "https://bon_appetit": 1.0,
    "https://hebbarskitchen": 1.0,
    "https://delish": 1.0
  },
  # 19: food festival events with jams, spices etc. in my area in the next week
  "19": {
    "https://eventbrite": 1.0
  },
  # 20: Porcelain clay suitable for throwing delicate tea cups on the wheel
  "20": {
    "https://ceramicsupplypittsburgh.com": 1.0,
    "https://https://ceramic-supply-inc.myshopify.com": 1.0,
    "https://standardclay.com": 1.0,
    "https://seattle-pottery-supply.myshopify.com": 1.0,
    "https://thepotterscenter.com": 1.0,
    "https://ceramic-supply-inc.myshopify.com": 1.0,
    "https://https://ceramic-supply-chicago.myshopify.com": 1.0,
    "https://bigceramicstore-com.myshopify.com": 1.0,
    "https://mysterycreekceramics.co.nz": 1.0,
    "https://ceramic-supply-chicago.myshopify.com": 1.0
  },
  # 21: High quality saffron threads for paella and risotto dishes
  "21": {
    "https://souschef.co.uk": 1.0,
    "https://worldspicemerchants.myshopify.com": 1.0,
    "https://thespiceway.com": 0.5,
    "https://https://especias-del-sol.myshopify.com": 1.0,
    "https://vanillabeankings.com": 1.0,
    "https://aromaticspices.myshopify.com": 1.0,
    "https://https://tm-shopify010-spice.myshopify.com": 1.0,
    "https://rumispice.com": 1.0,
    "https://oaktownspiceshop.com": 1.0,
    "https://alquimistes.myshopify.com": 1.0,
    "https://oldtownspices.com": 1.0,
    "https://diaspora-co-spices.myshopify.com": 1.0,
    "https://burlapandbarrel.com": 1.0,
    "https://https://universalfoodsqatar.myshopify.com": 1.0,
    "https://sabatonz.myshopify.com": 1.0,
    "https://gneissspice.com": 1.0,
    "https://especias-del-sol.myshopify.com": 1.0,
    "https://burlap-barrel.myshopify.com": 0.5,
    "https://theme124-spice-shop.myshopify.com": 0.5,
    "https://the-spice-guy.myshopify.com": 0.5,
    "https://old-town-spice-shop.myshopify.com": 1.0,
    "https://spiceology.com": 0.5,
    "https://savoryspiceshop.com": 0.5,
    "https://makkah-market.myshopify.com": 1.0,
    "https://aafricanspices.myshopify.com": 1.0
  },
  # 22: Bone broth in a pasta soup recipes
  "22": {
    "https://delish": 1.0,
    "https://bon_appetit": 1.0,
    "https://omnivorebooks.myshopify.com": 0.5,
    "https://seriouseats": 1.0,
    "https://nytimes": 0.5,
    "https://chefd.com": 0.5,
    "https://mediterranean_dish": 1.0
  },
  # 23: Handwoven baskets from natural materials like sweetgrass and willow
  "23": {
    "https://https://mexico-by-hand.myshopify.com": 1.0,
    "https://https://pauls-supplies.myshopify.com": 1.0,
    "https://https://connectedgoods.myshopify.com": 1.0,
    "https://https://notahdineh.myshopify.com": 1.0,
    "https://https://blinknow-foundation.myshopify.com": 1.0,
    "https://https://mappandthread.myshopify.com": 1.0,
    "https://https://na7xh7-9u.myshopify.com": 1.0
  },
  # 24: japanese dishes with rice and fish with fast cooking time
  "24": {
    "https://omnivorebooks.myshopify.com": 0.5,
    "https://delish": 1.0,
    "https://bon_appetit": 1.0,
    "https://chefd.com": 0.5,
    "https://nytimes": 0.5,
    "https://seriouseats": 1.0
  }
}
RATINGS_JSON_PATH = None  # change or set to None if pasting RATINGS_JSON above

# ---- 2) URL normalization helpers ----
def fix_double_scheme(u: str) -> str:
    # collapse "https://https://" pattern
    return re.sub(r"^(https?://)+(https?://)", r"\2", u.strip())

def strip_trailing_slash(u: str) -> str:
    return u[:-1] if u.endswith("/") else u

def ensure_scheme(u: str) -> str:
    # If it looks like a domain/path but no scheme, add https://
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", u):
        return u
    # Heuristic: has a dot or looks like a shopify domain -> add https://
    if "." in u or "myshopify" in u:
        return "https://" + u
    return u  # leave as-is; will likely fail to map and assert later

def canonicalize_url(u: str) -> str:
    return u
    # 1) quick obvious fixes
    u = fix_double_scheme(u)
    u = strip_trailing_slash(u)

    # 2) add scheme if needed
    u = ensure_scheme(u)

    # 3) parse & canonicalize host
    try:
        p = urlparse(u)
    except Exception:
        return u

    netloc = p.netloc.lower()
    # drop default ports
    netloc = re.sub(r":(80|443)$", "", netloc)
    # strip leading www.
    if netloc.startswith("www."):
        netloc = netloc[4:]

    # rebuild
    p = p._replace(netloc=netloc)
    canon = urlunparse(p)
    canon = strip_trailing_slash(canon)
    return canon

# ---- 3) Load the corpus config and build URL -> corpus_id mapping ----
def load_corpus_url_map(repo_id: str) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    """
    Returns:
      by_full_url: exact normalized URL -> corpus_id
      by_hostname: hostname -> [corpus_ids] (in case you want to experiment)
    """
    # We don't know the split name a priori. Try common options, then fallback.
    ds = load_dataset(repo_id, name="corpus")
    if isinstance(ds, DatasetDict):
        # prefer explicit splits if present
        for candidate in ["train", "corpus", "all", "validation", "test"]:
            if candidate in ds:
                corpus = ds[candidate]
                break
        else:
            # take the first split if unknown
            first_split = list(ds.keys())[0]
            corpus = ds[first_split]
    else:
        corpus = ds

    required_cols = {"url", "corpus_id"}
    missing = required_cols - set(corpus.column_names)
    if missing:
        raise RuntimeError(f"Corpus is missing required columns: {missing}")

    by_full_url: Dict[str, int] = {}
    by_hostname: Dict[str, List[int]] = {}

    for url, cid in zip(corpus["url"], corpus["corpus_id"]):
        if url is None:
            continue
        cu = canonicalize_url(str(url))
        by_full_url[cu] = int(cid)
        try:
            host = urlparse(cu).netloc
        except Exception:
            host = ""
        if host:
            by_hostname.setdefault(host, []).append(int(cid))

    return by_full_url, by_hostname

# ---- 4) Map a rated URL to a corpus_id (strict) ----
def map_url_to_corpus_id(u_raw: str, by_full_url: Dict[str, int]) -> int:
    u = canonicalize_url(u_raw)

    # Try exact normalized URL
    if u in by_full_url:
        return by_full_url[u]

    # Try again after forcing https (some corpora store only https)
    if u.startswith("http://"):
        u_https = "https://" + u[len("http://"):]
        u_https = strip_trailing_slash(u_https)
        if u_https in by_full_url:
            return by_full_url[u_https]

    # If still missing, raise to honor the "assert if not found" requirement
    raise AssertionError(f"Could not map rated URL to corpus_id: {u_raw!r} -> normalized {u!r}")

# ---- 5) Build rows for the new config ----
def build_achint_qrels_rows(
    ratings: Dict[str, Dict[str, float]],
    by_full_url: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows = []
    for qid_str, url_score_map in ratings.items():
        qid = int(qid_str)  # note: incoming is string; dataset uses int
        pos_all: List[int] = []
        pos_critical: List[int] = []

        # Deduplicate URLs per query by normalized form
        seen_norm = set()

        for url_raw, score in url_score_map.items():
            # skip null/empty
            if not url_raw or (isinstance(url_raw, str) and url_raw.strip() == ""):
                continue

            cu = canonicalize_url(url_raw)
            if cu in seen_norm:
                continue
            seen_norm.add(cu)

            if score is None:
                continue
            try:
                score_val = float(score)
            except Exception:
                continue

            if score_val > 0.0:
                cid = map_url_to_corpus_id(url_raw, by_full_url)
                pos_all.append(cid)
                if score_val >= 1.0:
                    pos_critical.append(cid)

        # Sort for determinism
        pos_all = sorted(set(pos_all))
        pos_critical = sorted(set(pos_critical))

        rows.append(
            {
                "query_id": qid,
                "positive_corpus_ids_all": pos_all,
                "positive_corpus_ids_critical_only": pos_critical,
            }
        )

    # sort rows by query_id to keep it tidy
    rows.sort(key=lambda r: r["query_id"])
    return rows

# ---- 6) Push to Hub as a new private config ----
def push_config(repo_id: str, config_name: str, split_name: str, rows: List[Dict[str, Any]], private: bool = True, token: str = None):
    ds = Dataset.from_list(rows)
    dsd = DatasetDict({split_name: ds})
    # This pushes a new config within the same dataset repo.
    dsd.push_to_hub(
        repo_id=repo_id,
        config_name=config_name,
        private=private,
        token=token,
    )
    print(f"Pushed {len(rows)} rows to {repo_id} (config={config_name!r}, split={split_name!r}, private={private}).")

# ---- 7) Main flow ----
def main():
    # Load ratings
    if RATINGS_JSON is None:
        if not RATINGS_JSON_PATH or not os.path.exists(RATINGS_JSON_PATH):
            raise FileNotFoundError("Provide RATINGS_JSON or set RATINGS_JSON_PATH to a valid JSON file.")
        with open(RATINGS_JSON_PATH, "r") as f:
            ratings = json.load(f)
    else:
        ratings = RATINGS_JSON

    # Build URL maps
    by_full_url, _by_host = load_corpus_url_map(REPO_ID)

    # Build new rows
    rows = build_achint_qrels_rows(ratings, by_full_url)

    # Optional: sanity check that every provided query_id is represented
    provided_qids = set(int(k) for k in ratings.keys())
    built_qids = set(r["query_id"] for r in rows)
    assert provided_qids == built_qids, f"Mismatch in qids: provided={sorted(provided_qids)} built={sorted(built_qids)}"

    # Push
    hf_token = os.environ.get("HF_TOKEN", None)  # or set manually
    push_config(REPO_ID, TARGET_CONFIG, TARGET_SPLIT, rows, private=True, token=hf_token)

if __name__ == "__main__":
    main()