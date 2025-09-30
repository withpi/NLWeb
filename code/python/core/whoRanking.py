# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Simplified WHO ranking for site selection.
"""

import os
from core.utils.utils import log
from core.llm import ask_llm
import asyncio
import json
from core.utils.json_utils import trim_json
from misc.logger.logging_config_helper import get_configured_logger
from core.schemas import create_assistant_result
import httpx
from typing import Any


logger = get_configured_logger("who_ranking_engine")


class WhoRanking:

    EARLY_SEND_THRESHOLD = 59
    NUM_RESULTS_TO_SEND = 10

    def __init__(self, handler, items, level="low"):
        logger.info(f"Initializing WHO Ranking with {len(items)} items")
        self.handler = handler
        self.level = level  # default to high level for WHO ranking
        self.items = items
        self.num_results_sent = 0
        self.rankedAnswers = []
        self.usePi = os.environ.get("WITHPI_API_KEY", "") != ""
        self.client = httpx.AsyncClient(timeout=10.0)
        self.scoreThreshold = 20 if self.usePi else 50

    async def piScoreItem(self, description: str) -> int:
        resp = await self.client.post(
            "https://api.withpi.ai/v1/scoring_system/score",
            headers={
                "x-api-key": os.environ.get("WITHPI_API_KEY", ""),
                "x-model-override": "pi-scorer-bert:modal:https://pilabs--pi-modelserver-scorermodel-invocations.modal.run",
            },
            json={
                "llm_input": self.handler.query,
                "llm_output": description,
                "scoring_spec": [
                    {"question": "Is the response relevant to the input?"}
                ],
            },
        )
        resp.raise_for_status()
        score_result = resp.json()
        return int(score_result["total_score"] * 100)

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

    def get_ranking_prompt(self, query, site_description):
        """Construct the WHO ranking prompt with the given query and site description."""
        prompt = f"""Assign a score between 0 and 100 to the following site based 
        the likelihood that the site will contain an answer to the user's question.
        If the user is looking to buy a product, the site should sell the product, not 
        just have useful information. 

The user's question is: {query}

The site's description is: {site_description}
"""

        response_structure = {
            "score": "integer between 0 and 100",
            "description": "short description of why this site is relevant",
            "query": "the optimized query to send to this site (only if score > 70)",
        }

        return prompt, response_structure

    async def rankItem(self, url, json_str, name, site):
        """Rank a single site for relevance to the query."""
        try:
            description = trim_json(json_str)
            if self.usePi:
                pi_score = await self.piScoreItem(str(description))
                ranking = {
                    "score": pi_score,
                    "description": self.getDescription(description),
                }
            else:
                prompt, ans_struc = self.get_ranking_prompt(
                    self.handler.query, description
                )
                ranking = await ask_llm(
                    prompt,
                    ans_struc,
                    level=self.level,
                    query_params=self.handler.query_params,
                    timeout=90,
                )

            # Ensure ranking has required fields (handle LLM failures/timeouts)
            if not ranking or not isinstance(ranking, dict):
                ranking = {
                    "score": 0,
                    "description": "Failed to rank",
                    "query": self.handler.query,
                }
            if "score" not in ranking:
                ranking["score"] = 0
            if "query" not in ranking:
                ranking["query"] = self.handler.query

            # Log the LLM score
            # LLM Score recorded

            # Handle both string and dictionary inputs for json_str
            schema_object = (
                json_str if isinstance(json_str, dict) else json.loads(json_str)
            )

            # Store the result
            ansr = {
                "url": url,
                "site": site,
                "name": name,
                "ranking": ranking,
                "schema_object": schema_object,
                "sent": False,
            }

            # Send immediately if high score
            if ranking.get("score", 0) > self.EARLY_SEND_THRESHOLD:
                logger.info(
                    f"High score site: {name} (score: {ranking['score']}) - sending early"
                )
                await self.sendAnswers([ansr])

            self.rankedAnswers.append(ansr)
            logger.error(f"Site {name} added to ranked answers")

        except Exception as e:
            logger.error(f"Error in rankItem for {name}: {str(e)}")
            logger.debug(f"Full error trace: ", exc_info=True)
            # Still add the item with a zero score so we don't lose it completely
            try:
                schema_object = (
                    json_str if isinstance(json_str, dict) else json.loads(json_str)
                )
                ansr = {
                    "url": url,
                    "site": site,
                    "name": name,
                    "ranking": {
                        "score": 0,
                        "description": f"Error: {str(e)}",
                        "query": self.handler.query,
                    },
                    "schema_object": schema_object,
                    "sent": False,
                }
                self.rankedAnswers.append(ansr)
            except:
                pass  # Skip this item entirely if we can't even create a basic record

    async def sendAnswers(self, answers, force=False):
        """Send ranked sites to the client."""
        json_results = []

        for result in answers:
            # Stop if we've already sent enough
            if self.num_results_sent + len(json_results) >= self.NUM_RESULTS_TO_SEND:
                logger.error(
                    f"Stopping at {len(json_results)} results to avoid exceeding limit of {self.NUM_RESULTS_TO_SEND}"
                )
                break

            # Extract site type from schema_object
            schema_obj = result.get("schema_object", {})
            site_type = schema_obj.get("@type", "Website")

            result_item = {
                "@type": site_type,  # Use the actual site type
                "url": result["url"],
                "name": result["name"],
                "score": result["ranking"]["score"],
            }

            # Include description if available
            if "description" in result["ranking"]:
                result_item["description"] = result["ranking"]["description"]

            # Always include query field (required for WHO ranking)
            if "query" in result["ranking"]:
                result_item["query"] = result["ranking"]["query"]
            else:
                # Fallback to original query if no custom query provided
                result_item["query"] = self.handler.query

            json_results.append(result_item)
            result["sent"] = True

        logger.error(f"Sending {len(json_results)} json_results")

        if json_results:
            # Use the new schema to create and auto-send the message
            create_assistant_result(json_results, handler=self.handler)
            self.num_results_sent += len(json_results)
            logger.error(
                f"Sent {len(json_results)} results, total sent: {self.num_results_sent}/{self.NUM_RESULTS_TO_SEND}"
            )

    async def do(self):
        """Main execution method - rank all sites concurrently."""
        
        # Create tasks for all sites
        tasks = []
        for url, json_str, name, site in self.items:
            tasks.append(asyncio.create_task(self.rankItem(url, json_str, name, site)))

        # Wait for all ranking tasks to complete
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during ranking tasks: {str(e)}")

        # Filter and sort final results
        filtered = [r for r in self.rankedAnswers if r.get('ranking', {}).get('score', 0) > self.scoreThreshold]
        ranked = sorted(filtered, key=lambda x: x.get('ranking', {}).get("score", 0), reverse=True)
        self.handler.final_ranked_answers = ranked[:self.NUM_RESULTS_TO_SEND]
        
        print(f"\n=== WHO RANKING: Filtered to {len(filtered)} results with score > {self.scoreThreshold} ===")
        
        # Print the ranked sites with scores
        print("\nRanked sites (top 10):")
        for i, r in enumerate(ranked[:self.NUM_RESULTS_TO_SEND], 1):
            score = r.get('ranking', {}).get('score', 0)
            print(f"  {i}. {r['name']} - Score: {score}")
        print("=" * 60)
        
        # Final ranked results processed

        # Send any remaining results that haven't been sent
        results_to_send = [r for r in ranked if not r["sent"]][
            : self.NUM_RESULTS_TO_SEND - self.num_results_sent
        ]

        logger.error(
            f"NUM_RESULTS_TO_SEND2={self.NUM_RESULTS_TO_SEND} -- {self.num_results_sent} -- {len(results_to_send)}"
        )

        if results_to_send:
            logger.error(f"Sending final batch of {len(results_to_send)} results")
            await self.sendAnswers(results_to_send, force=True)
