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

# Who handler is work in progress for answering questions about who
# might be able to answer a given query

logger = get_configured_logger("who_handler")

DEFAULT_NLWEB_ENDPOINT = "https://nlwm.azurewebsites.net/ask"


class WhoHandler(NLWebHandler):

    def __init__(self, query_params, http_handler):
        self.client = httpx.AsyncClient(timeout=10.0)
        # Remove site parameter - we'll use nlweb_sites
        if "site" in query_params:
            del query_params["site"]

        # Keep prev_queries if provided for context, but don't use 'prev' format
        # The who handler can use previous queries to understand follow-up questions
        if "prev" in query_params:
            del query_params["prev"]
        # Keep prev_queries for context if provided
        super().__init__(query_params, http_handler)

        self.query_classification_threshold = 0.4

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

    async def crossEncodeItems(self, descriptions: list[str]) -> list[float]:
        try:
            formatted_descriptions = [
                json.dumps(json.loads(description), indent=2, ensure_ascii=False)
                for description in descriptions
            ]
        except:
            formatted_descriptions = descriptions

        resp = await self.client.post(
            "https://api.withpi.ai/v1/search/query_to_passage/score",
            headers={
                "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
                "x-hotswaps": "pi-cross-encoder-small:pi-cross-encoder-qwen",
                "x-model-override": "pi-cross-encoder-qwen:modal:https://pilabs-qwen--pi-modelserver-scorermodel-invocations.modal.run",
            },
            json={
                "query": self.query,
                "passages": formatted_descriptions,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def crossEncodeItem(self, formatted_description: str) -> float:
        resp = await self.client.post(
            "https://api.withpi.ai/v1/search/query_to_passage/score",
            headers={
                "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
                "x-hotswaps": "pi-cross-encoder-small:pi-cross-encoder-qwen",
                "x-model-override": "pi-cross-encoder-qwen:modal:https://pilabs-qwen--pi-modelserver-scorermodel-invocations.modal.run",
            },
            json={
                "query": self.query,
                "passages": [formatted_description],
            },
        )
        resp.raise_for_status()
        return resp.json()[0]

    async def crossEncodeItemUsingPiScore(self, formatted_description: str) -> float:
        resp = await self.client.post(
            "https://api.withpi.ai/v1/scoring_system/score",
            headers={
                "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
                "x-model-override": "pi-scorer-nlweb-who:modal:https://pilabs-nlweb--pi-modelserver-scorermodel-invocations.modal.run",
            },
            json={
                "llm_input": self.query,
                "llm_output": formatted_description,
                "scoring_spec": [
                    {
                        "question": "Is the response relevant to the input?",
                        "label": "Relevance",
                        "weight": 1.0,
                    }
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["total_score"]

    async def piScoreItem(self, description: str, query_annotations) -> int:
        try:
            formatted_description = json.dumps(
                json.loads(description), indent=2, ensure_ascii=False
            )
        except:
            formatted_description = description

        # ce_score = await self.crossEncodeItem(formatted_description)
        ce_score = await self.crossEncodeItemUsingPiScore(formatted_description)

        scoring_spec = []
        for category, score in query_annotations.items():
            if score > self.query_classification_threshold:
                scoring_spec.append(
                    {
                        "question": f"Is the response about {category}?",
                        "label": category,
                        "weight": 1.0,
                    }
                )
        if len(scoring_spec) > 0:
            resp = await self.client.post(
                "https://api.withpi.ai/v1/scoring_system/score",
                headers={
                    "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
                    "x-model-override": "pi-scorer-nlweb-who:modal:https://pilabs-nlweb--pi-modelserver-scorermodel-invocations.modal.run",
                },
                json={
                    # "llm_input": self.query,
                    "llm_input": "",
                    "llm_output": formatted_description,
                    "scoring_spec": scoring_spec,
                },
            )
            resp.raise_for_status()
            pi_score = resp.json()["total_score"]

            ce_weight = 0.4
            return int(((pi_score + ce_weight * ce_score) / (1.0 + ce_weight)) * 100)
        else:
            return ce_score * 100

    async def queryClassify(self, scoring_spec) -> dict:
        resp = await self.client.post(
            "https://api.withpi.ai/v1/scoring_system/score",
            headers={
                "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
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
                return self.getFirst(schema_org["description"])
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

    async def rankItem(
        self, url, json_str, name, site, categories: list[str], query_annotations
    ):
        """Rank a single site for relevance to the query."""
        description = trim_json(json_str)
        pi_score = await self.piScoreItem(str(description), query_annotations)
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
            "travels",
            "movies",
            "events",
            "education",
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

        query_annotations_trigger = await self.queryClassify(
            scoring_spec=[
                {
                    "question": f"Is the user's query seeking to buy something?",
                    "label": "shopping_intent",
                    "weight": 1.0,
                }
            ]
        )
        query_annotations_trigger = query_annotations_trigger["question_scores"]
        print(f"Query triggering: {query_annotations_trigger}")
        is_shopping = (
            query_annotations_trigger["shopping_intent"]
            > self.query_classification_threshold
        )
        # CATEGORIES = SHOPPING_CATEGORIES if is_shopping else OTHER_CATEGORIES
        # print(f"Using categories: {CATEGORIES}")
        CATEGORIES = ALL_CATEGORIES

        # TODO: condition on shopping or not shopping
        query_annotations = await self.queryClassify(
            scoring_spec=[
                {
                    "question": f"Is the query seeking information about {category} category?",
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

        # If there are more than 3 above threshold, keep them all;
        # otherwise, take top 3 overall.
        if len(above_thresh) > 3:
            search_cats = above_thresh
        else:
            search_cats = [cat for cat, _ in sorted_items[:3]]

        # Search using the special nlweb_sites collection
        # search_cats = [
        #        category for category, score in query_annotations.items() if score > 0.5
        #    ]
        k = 60 if "num" not in self.query_params else int(self.query_params["num"])
        print(f"Search cat scores: {query_annotations}")
        print(f"Search cats: {search_cats}")
        print(f"Search k: {k}")
        BASE_CATEGORIES = ["ALL"]  # if is_shopping else ["NON_SHOPIFY"]
        items = await LOCAL_CORPUS.search(
            query=str(self.query),
            categories=["ALL", "NON_SHOPIFY"],
            # categories=BASE_CATEGORIES + search_cats,
            k=k,
            # k=20
        )

        # self.final_retrieved_items = items
        print(f"\n=== WHO HANDLER: Retrieved {len(items)} items from nlweb_sites ===")

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for url, json_str, name, site, categories in items:
                tasks.append(
                    tg.create_task(
                        self.rankItem(
                            url, json_str, name, site, categories, query_annotations
                        )
                    )
                )

        # Use min_score from handler if available, otherwise default to 51
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
            f"\n=== WHO RANKING: Filtered to {len(filtered)} results with score > {min_score_threshold} ==="
        )

        for i in range(0, len(to_send), 3):
            subset = to_send[i:i+3]
            string = ""
            for item in subset:
                string += f"{item['url']:<60}"
            print(string)


        await self.sendAnswers(to_send, force=True)
        return self.return_value
