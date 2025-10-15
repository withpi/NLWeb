import json
import os
from core.baseHandler import NLWebHandler
import httpx
from typing import Any
from misc.logger.logging_config_helper import get_configured_logger
import asyncio
from core.local_corpus import LOCAL_CORPUS
from core.schemas import create_assistant_result
from core.utils.json_utils import trim_json
import numpy as np

# Who handler is work in progress for answering questions about who
# might be able to answer a given query

logger = get_configured_logger("who_handler")

DEFAULT_NLWEB_ENDPOINT = "https://nlwm.azurewebsites.net/ask"

def auto_score_cutoff(
    scores,
    smooth: int = 3,
    min_keep: int = 4,
    drop_ratio: float = 2.0,
    verbose: bool = False,
):
    """
    Given a list of descending scores (0–100), pick a cutoff threshold based on relative drop size.

    Args:
        scores: list of numeric scores, ideally sorted descending.
        smooth: optional window for smoothing drops (1 = no smoothing).
        min_keep: minimum number of results to always keep (avoid cutting too early).
        drop_ratio: how much larger a drop must be (vs median of smaller drops) to count as a "big drop".
        verbose: print diagnostics if True.

    Returns:
        cutoff_score (float), cutoff_index (int)
    """
    if len(scores) <= min_keep:
        return min(scores), len(scores)

    scores = np.array(sorted(scores, reverse=True))
    diffs = np.diff(scores) * -1  # positive values are drops

    # Optional smoothing of differences
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        diffs = np.convolve(diffs, kernel, mode="same")

    # Typical (small) drop magnitude
    typical_drop = np.median(diffs[:max(min_keep, len(diffs)//3)])
    big_drops = np.where(diffs > drop_ratio * typical_drop)[0]

    if len(big_drops) == 0:
        cutoff_idx = len(scores)
    else:
        cutoff_idx = max(min_keep, big_drops[0] + 1)

    cutoff_score = scores[cutoff_idx - 1]

    if verbose:
        print(f"Typical drop={typical_drop:.3f}, chosen cutoff at index {cutoff_idx}, score {cutoff_score:.3f}")

    return cutoff_score, cutoff_idx

class WhoHandler(NLWebHandler):

    def __init__(self, query_params, http_handler, client: httpx.AsyncClient):
        self.client = client
        # Remove site parameter - we'll use nlweb_sites
        if "site" in query_params:
            del query_params["site"]

        # Keep prev_queries if provided for context, but don't use 'prev' format
        # The who handler can use previous queries to understand follow-up questions
        if "prev" in query_params:
            del query_params["prev"]
        # Keep prev_queries for context if provided
        super().__init__(query_params, http_handler)

        self.query_classification_threshold = 0.35

    def _build_nlweb_url(self, site_url, site_type=None):
        """Helper function to build the complete NLWEB URL with all parameters."""
        from urllib.parse import quote

        params = []
        params.append(f"site={site_url}")

        # Add the user's query
        if self.query:
            params.append(f"query={quote(self.query)}")

        # Check if it's a Shopify site and add db parameter
        if site_type in ["ShopifyStore", "Shopify"] or "shopify" in site_url.lower():
            params.append("db=shopify_mcp")

        # Add tool parameter to go directly to search
        params.append("tool=search")

        # Construct the full URL
        return f"{DEFAULT_NLWEB_ENDPOINT}?{'&'.join(params)}"

    async def send_message(self, message):
        """Override send_message to ensure URLs point to /ask endpoint with site parameter."""
        # Check if message contains results with URLs
        if isinstance(message, dict):
            # Handle messages with 'content' field (results)
            if "content" in message and isinstance(message["content"], list):
                for result in message["content"]:
                    if "url" in result:
                        url = result["url"]
                        # If URL doesn't start with http:// or https://, convert to /ask endpoint
                        if not url.startswith(("http://", "https://")):
                            site_type = result.get("@type", "")
                            result["url"] = self._build_nlweb_url(url, site_type)
                            logger.debug(
                                f"Modified URL from '{url}' to '{result['url']}'"
                            )

            # Handle single result messages
            elif "url" in message:
                url = message["url"]
                if not url.startswith(("http://", "https://")):
                    site_type = message.get("@type", "")
                    message["url"] = self._build_nlweb_url(url, site_type)
                    logger.debug(f"Modified URL from '{url}' to '{message['url']}'")

        # Call parent class's send_message with modified message
        await super().send_message(message)

    async def piScoreItem(self, description: str, query_annotations, url) -> int:
        try:
            formatted_description = json.dumps(
                json.loads(description), indent=2, ensure_ascii=False
            )
        except:
            formatted_description = description

        scoring_spec = [
            {
                "question": "Is the response relevant to the input?",
                "label": "Relevance",
                "weight": 1.0,
            }
        ]
        is_vertical_query = False
        is_shopping_doc = "myshopify.com" in url
        for category, score in query_annotations.items():
            if score > self.query_classification_threshold:
                is_vertical_query = True
                scoring_spec.append(
                    {
                        "question": f"Is the response about {category} or a {category} website?",
                        "label": category,
                        "weight": 1.0,
                    }
                )
        if is_vertical_query:
            scoring_spec.append(
                {
                    "question": f"Is this website a store that sells products?",
                    "label": "is_store",
                    "weight": 0.0,
                }
            )
        else:
            scoring_spec.append(
                {
                    "question": f"Is the shop relevant to the query?",
                    "label": "Shop Relevance",
                    "weight": 0.25,
                }
            )
            scoring_spec.append(
                {
                    "question": f"Does the shop sell the product explicitly mentioned in the query?",
                    "label": "Product Presence",
                    "weight": 0.25,
                }
            )	
            scoring_spec.append(
                {
                    "question": f"Does the shop sell the equipments that are being asked for in the query?",
                    "label": "Equipment Presence",
                    "weight": 0.25,
                }
            )	
            scoring_spec.append(
                {
                    "question": f"Are the products appropriate for the audience mentioned in the query?",
                    "label": "Audience Appropriateness",
                    "weight": 0.25,
                }
            )
            	
        if is_shopping_doc and is_vertical_query:
            # filter doc
            print(f"{url} filtering 1")
            return 0
        resp = await self.client.post(
            "https://api.withpi.ai/v1/scoring_system/score",
            headers={
                "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
                "x-hotswaps": "pi-scorer-bert:pi-scorer-nlweb-who",
                "x-model-override": "pi-scorer-nlweb-who:modal:https://pilabs-nlweb--pi-modelserver-scorermodel-invocations.modal.run",
                #"x-model-override": "pi-scorer-bert:modal:https://pilabs-nlweb--pi-modelserver-scorermodel-invocations.modal.run",
            },
            json={
                "llm_input": self.query,
                "llm_output": formatted_description,
                "scoring_spec": scoring_spec,
            },
        )
        resp.raise_for_status()
        question_scores = resp.json()["question_scores"]
        if is_vertical_query:
            if question_scores["is_store"] > 0.25:
                print(f"{url}  filtering 2 {question_scores["is_store"]}")
                return 0
            for category, score in query_annotations.items():
                if score > self.query_classification_threshold:
                    if question_scores[category] < 0.1:
                        print(f"{url}  filtering 3: {category} {question_scores[category]}")
                        return 0
        ce_score = question_scores.pop("Relevance", 0.0)
        pi_score = (
            sum(question_scores.values()) / len(question_scores)
            if len(question_scores) > 0
            else None
        )

        if pi_score is not None:
            ce_weight = 0.4
            print(f"SCORE DEBUG =============\nurl={url}, pi_score={pi_score}, ce_weight={ce_weight}, intent_scores={question_scores}")
            return int(((pi_score + ce_weight * ce_score) / (1.0 + ce_weight)) * 100)
        return int(ce_score * 100)

    async def queryClassify(self, scoring_spec) -> dict:
        resp = await self.client.post(
            "https://api.withpi.ai/v1/scoring_system/score",
            headers={
                "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
                "x-model-override": "pi-scorer-bert:modal:https://pilabs-nlweb--pi-modelserver-scorermodel-invocations.modal.run",
                # "x-model-override": "pi-scorer-nlweb-who:modal:https://pilabs-nlweb--pi-modelserver-scorermodel-invocations.modal.run",
            },
            json={
                "llm_input": "",
                "llm_output": json.dumps({"query": self.query}, indent=2),
                "scoring_spec": scoring_spec,
            },
        )
        resp.raise_for_status()
        score_result = resp.json()

        print(f"QUERY CLASSIFICATION: {score_result=}")

        return score_result

    def getFirst(self, field: list[str] | str) -> str:
        if isinstance(field, list):
            if len(field) > 0:
                return field[0]
            else:
                return ""
        elif isinstance(field, str):
            return field
        else:
            return str(field)

    def getDescription(self, schema_org: dict[str, Any] | str) -> str:
        if isinstance(schema_org, dict):
            if "description" in schema_org:
                # logger.error("Description found: %s", schema_org["description"])
                return "{} ...".format(
                    self.getFirst(schema_org["description"])
                    .removeprefix("## PRODUCTS FOUND")
                    .strip()[:200]
                )
            elif "text" in schema_org:
                return self.getFirst(schema_org["text"])
            elif "name" in schema_org:
                return self.getFirst(schema_org["name"])
            else:
                return json.dumps(schema_org)  # Fallback to full JSON string
        elif isinstance(schema_org, str):
            return schema_org
        else:
            return str(schema_org)

    async def rankItemWithRetries(self, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.rankItem(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in rankItem (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise

    async def rankItem(
        self, url, json_str, name, site, categories: list[str], query_annotations
    ):
        """Rank a single site for relevance to the query."""
        description = trim_json(json_str)
        pi_score = await self.piScoreItem(str(description), query_annotations, url)
        ranking = {
            "score": pi_score,
            "description": self.getDescription(description),
            "query": self.query,
        }

        # Handle both string and dictionary inputs for json_str
        schema_object = json_str if isinstance(json_str, dict) else json.loads(json_str)

        # Add WHO classification categories to schema_object (list of all categories that found this doc)
        # This is just to help the benchmark understand the underlying classification.
        schema_object["who_categories"] = categories

        # Store the result
        return {
            "url": url,
            "site": site,
            "name": name,
            "ranking": ranking,
            "schema_object": schema_object,
            "categories": categories,
            "sent": False,
        }

    async def sendAnswers(self, answers, force=False):
        """Send ranked sites to the client."""
        json_results = []

        for result in answers:
            # Extract site type from schema_object
            schema_obj = result.get("schema_object", {})
            site_type = schema_obj.get("@type", "Website")

            result_item = {
                "@type": site_type,  # Use the actual site type
                "url": result["url"],
                "name": result["name"],
                "score": result["ranking"]["score"],
                "schema_object": json.dumps(
                    schema_obj
                ),  # Include full schema_object with who_category
            }

            # Include description if available
            if "description" in result["ranking"]:
                result_item["description"] = result["ranking"]["description"]

            # Always include query field (required for WHO ranking)
            if "query" in result["ranking"]:
                result_item["query"] = result["ranking"]["query"]
            else:
                # Fallback to original query if no custom query provided
                result_item["query"] = self.query

            json_results.append(result_item)
            result["sent"] = True

        logger.error(f"Sending {len(json_results)} json_results")

        if json_results:
            # Use the new schema to create and auto-send the message
            await create_assistant_result(json_results, handler=self)
            logger.error(f"Sent {len(json_results)} results")

    async def getQueryAnnotationsWithRetries(self, CATEGORIES):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.getQueryAnnotations(CATEGORIES)
            except Exception as e:
                logger.error(
                    f"Error in getQueryAnnotations (attempt {attempt + 1}): {e}"
                )
                if attempt == max_retries - 1:
                    raise

    async def getQueryAnnotations(self, CATEGORIES):
        # TODO: condition on shopping or not shopping
        query_annotations = await self.queryClassify(
            scoring_spec=[
                {
                    "question": f"Is the query's primary intent to find {category}?",
                    "label": category,
                    "weight": 1.0,
                }
                for category in CATEGORIES
            ]
        )
        query_annotations = query_annotations["question_scores"]

        # Sort categories by descending score
        sorted_items = sorted(
            query_annotations.items(), key=lambda x: x[1], reverse=True
        )

        # Get all above 0.1
        above_thresh = [
            cat
            for cat, score in sorted_items
            if score > self.query_classification_threshold
        ]

        print(f"above thresh: {above_thresh}")

        # If there are more than 3 above threshold, keep them all;
        # otherwise, take top 3 overall.
        # if len(above_thresh) > 3:
        #     search_cats = above_thresh
        # else:
        #     search_cats = [cat for cat, _ in sorted_items[:3]]
        # print(f"Search cat scores: {query_annotations}")
        # print(f"Search cats: {search_cats}")
        return query_annotations

    async def runQuery(self):
        # Always use general search with nlweb_sites
        logger.info("Using general search method with site=nlweb_sites for who query")

        # CATEGORIES = [
        #     "Food & Beverage",
        #     "Health & Wellness",
        #     "Ceramics & Pottery",
        #     "Home & Lifestyle",
        #     "Textiles & Crafts",
        #     "Services & Education",
        #     "Books & Media",
        #     "Kitchen & Culinary Tools",
        #     "Sports & Outdoor",
        #     "Apparel & Accessories",
        #     "General & Misc",
        #     "Garden & Botanicals",
        #     "Bags & Leather",
        #     "Jewelry & Watches",
        #     "Toys, Hobbies & Collectibles",
        # ]
        ALL_CATEGORIES = [
            "recipes",
            "travel/tourism, sightseeing, or things to do",
            "movies",
            "events",
            "educational content",
            "podcasts",
        ]
        # ALL_CATEGORIES = [
        #     "Travel & Events",
        #     "Bags & Leather",
        #     "Kitchen & Culinary Tools",
        #     "Japanese Culture & Goods",
        #     "Restaurants & Dining",
        #     "Home & Garden",
        #     "Beverages",
        #     "Education & Research",
        #     "Textiles & Crafts",
        #     "Apparel & Accessories",
        #     "Cooking & Food Media",
        #     "Ceramics & Pottery",
        #     "Health & Wellness",
        #     "Home & Lifestyle",
        #     "Pets & Aquatics",
        #     "Real Estate",
        #     "Baby & Kids",
        #     "Meals & Nutrition",
        #     "Sports & Outdoor",
        #     "General & Misc",
        #     "Crafts & Sewing",
        #     "Toys, Hobbies & Collectibles",
        #     "Jewelry & Watches",
        #     "Culinary Ingredients & Pantry",
        #     "Garden & Botanicals",
        #     "Nutrition & Supplements",
        #     "Beauty & Personal Care",
        #     "Food & Beverage",
        #     "Business & Industrial",
        #     "Apparel & Lifestyle",
        #     "Kitchen Tools & Utensils",
        #     "Media & Entertainment",
        #     "Packaged Foods & Snacks",
        #     "Books & Media",
        #     "Smoke & Vape",
        #     "Services & Education",
        #     "Coffee Equipment & Brewing",
        # ]
        SHOPPING_CATEGORIES = [
            "Bags & Leather",
            "Kitchen & Culinary Tools",
            "Textiles & Crafts",
            "Apparel & Accessories",
            "Ceramics & Pottery",
            "Health & Wellness",
            "Home & Lifestyle",
            "Pets & Aquatics",
            "Baby & Kids",
            "Sports & Outdoor",
            "General & Misc",
            "Toys, Hobbies & Collectibles",
            "Jewelry & Watches",
            "Garden & Botanicals",
            "Beauty & Personal Care",
            "Food & Beverage",
            "Business & Industrial",
            "Books & Media",
            "Smoke & Vape",
            "Services & Education",
        ]
        OTHER_CATEGORIES = [
            "Meals & Nutrition",
            "Travel & Events",
            "Japanese Culture & Goods",
            "Beverages",
            "Apparel & Lifestyle",
            "Kitchen Tools & Utensils",
            "Education & Research",
            "Cooking & Food Media",
            "Media & Entertainment",
            "Ceramics & Pottery",
            "Crafts & Sewing",
            "Restaurants & Dining",
            "Packaged Foods & Snacks",
            "Real Estate",
            "Culinary Ingredients & Pantry",
            "Nutrition & Supplements",
            "Home & Garden",
            "Coffee Equipment & Brewing",
        ]

        # CATEGORIES = SHOPPING_CATEGORIES if is_shopping else OTHER_CATEGORIES
        # print(f"Using categories: {CATEGORIES}")
        CATEGORIES = ALL_CATEGORIES
        k = 60 if "num" not in self.query_params else int(self.query_params["num"])
        print(f"Search k: {k}")
        async with asyncio.TaskGroup() as tg:
            items_task = tg.create_task(
                LOCAL_CORPUS.searchWithRetries(
                    query=str(self.query),
                    categories=["ALL", "NON_SHOPIFY"],
                    # categories=BASE_CATEGORIES + search_cats,
                    k=k,
                    # k=20
                )
            )
            query_annotations_task = tg.create_task(
                self.getQueryAnnotationsWithRetries(CATEGORIES)
            )

        items = items_task.result()
        query_annotations = query_annotations_task.result()

        # self.final_retrieved_items = items
        print(f"\n=== WHO HANDLER: Retrieved {len(items)} items from nlweb_sites ===")

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for url, json_str, name, site, categories in items:
                tasks.append(
                    tg.create_task(
                        self.rankItemWithRetries(
                            url, json_str, name, site, categories, query_annotations
                        )
                    )
                )

        # Collect all scores from tasks (ignoring any missing/invalid)
        scores = [
            r.result().get("ranking", {}).get("score", 0)
            for r in tasks
            if r.result() and "ranking" in r.result()
        ]

        if scores:
            # Use automatic cutoff instead of static threshold
            min_score_threshold = auto_score_cutoff(scores, smooth=3, min_keep=5, drop_ratio=1.5)
            if isinstance(min_score_threshold, tuple):
                min_score_threshold = min_score_threshold[0]
        else:
            # fallback if no scores
            min_score_threshold = getattr(self, "min_score", 51)

        max_results = getattr(self, "max_results", 10)

        filtered = [
            r.result()
            for r in tasks
            if r.result().get("ranking", {}).get("score", 0) > min_score_threshold
        ]
        ranked = sorted(
            filtered, key=lambda x: x.get("ranking", {}).get("score", 0), reverse=True
        )
        to_send = ranked[:max_results]

        print(
            f"\n=== WHO RANKING [new]: Filtered to {len(filtered)} results with score > {min_score_threshold:.2f} ==="
        )

        for i in range(0, len(to_send), 3):
            subset = to_send[i : i + 3]
            string = ""
            for item in subset:
                string += f"{item['url']:<60}"
            print(string)

        await self.sendAnswers(to_send, force=True)
        return self.return_value
