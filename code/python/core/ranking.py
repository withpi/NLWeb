# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file contains the code for the ranking stage. 

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from core.utils.utils import log
from core.llm import ask_llm
import asyncio
import json
from core.utils.json_utils import trim_json
from core.prompts import find_prompt, fill_prompt
from misc.logger.logging_config_helper import get_configured_logger
from core.schemas import create_assistant_result, create_status_message, Message, SenderType, MessageType
from core.utils.utils import record_llm_call
logger = get_configured_logger("ranking_engine")


class Ranking:
     
    EARLY_SEND_THRESHOLD = 59
    NUM_RESULTS_TO_SEND = 10

    FAST_TRACK = 1
    REGULAR_TRACK = 2
    WHO_RANKING = 3
    CONVERSATION_SEARCH = 4

    # This is the default ranking prompt, in case, for some reason, we can't find the site_type.xml file.
    RANKING_PROMPT = ["""  Assign a score between 0 and 100 to the following {site.itemType}
based on how relevant it is to the user's question. Use your knowledge from other sources, about the item, to make a judgement. 
If the score is above 50, provide a short description of the item highlighting the relevance to the user's question, without mentioning the user's question.
Provide an explanation of the relevance of the item to the user's question, without mentioning the user's question or the score or explicitly mentioning the term relevance.
If the score is below 75, in the description, include the reason why it is still relevant.
The user's question is: {request.query}. The item's description is {item.description}""",
    {"score" : "integer between 0 and 100", 
 "description" : "short description of the item"}]
    
    WHO_RANKING_PROMPT = ["""  Assign a score between 0 and 100 to the following site
                          based on how relevant the site may be to answering the user's question.
                          The user's question is: {request.query}. The site's description is {item.description}
                          
                          Additionally, if this site is relevant (score > 50), provide an optimized query 
                          that should be sent to this specific site. The query should be tailored to:
                          - Match the site's specific domain and expertise
                          - Extract the most relevant information from that site
                          - Be more specific than the original query when appropriate
                          
                          For example, if the user asks "I need to make bread" and the site is about:
                          - Kitchen equipment: rewrite to "bread making equipment" or "bread baking tools"
                          - Ingredients/groceries: rewrite to "bread ingredients" or "flour yeast for bread"
                          - Recipes: keep as "bread recipes" or just "bread"
                          
                          If the original query is already well-suited for the site, use the original query.
                          """,
                            {"score" : "integer between 0 and 100", 
                            "description" : "short description of the item",
                            "query" : "the optimized query to send to this site (only if score > 50)"}]
    
    CONVERSATION_SEARCH_PROMPT = ["""Assign a score between 0 and 100 to the following past conversation
                          based on how relevant it is to the user's current search query.
                          Consider both the original question asked and the response received.
                          
                          The user's search query is: {request.query}
                          
                          Past conversation details:
                          {item.description}
                          
                          Score higher if:
                          - The conversation directly addresses the search topic
                          - The response contains useful information related to the query
                          - The conversation summary or topics match the search intent
                          
                          Provide a brief description highlighting why this conversation is relevant.""",
                          {"score": "integer between 0 and 100",
                           "description": "brief description of why this past conversation is relevant to the search"}]
    
    PRODUCT_FOCUSED_PROMPT = ["""Assign a score between 0 and 100 based on how well this product matches the user's search.
                          
                          Focus on product details in your description:
                          - Product name and brand (if available)
                          - Price or price range
                          - Key features or specifications
                          - Why this specific product matches the search
                          
                          Do NOT use phrases like "This webpage" or "This page" or "This site".
                          Instead, directly describe the product itself.
                          
                          Examples of good descriptions:
                          - "All-Clad D3 stainless steel frying pan, 12-inch, $149, triple-ply construction for even heating"
                          - "Le Creuset enameled cast iron Dutch oven in cherry red, 5.5 quart capacity, oven-safe to 500°F"
                          - "Breville Smart Oven Air Fryer, $399, 13 cooking functions including air fry, dehydrate, and slow cook"
                          
                          The user's search: {request.query}
                          Product information: {item.description}""",
                          {"score": "integer between 0 and 100",
                           "description": "product-focused description with brand, price, and key features"}]
 
    RANKING_PROMPT_NAME = "RankingPrompt"
     
    def get_ranking_prompt(self):
        site = self.handler.site
        item_type = self.handler.item_type
        
        # Check for special ranking types first
        if (self.ranking_type == Ranking.WHO_RANKING):
            return self.WHO_RANKING_PROMPT[0], self.WHO_RANKING_PROMPT[1]
        if (self.ranking_type == Ranking.CONVERSATION_SEARCH):
            return self.CONVERSATION_SEARCH_PROMPT[0], self.CONVERSATION_SEARCH_PROMPT[1]
        
        # Check if using Bing search or any e-commerce/product sites
        db_param = self.handler.query_params.get('db') if hasattr(self.handler, 'query_params') else None
        
        # Use product-focused prompt for Bing search or known e-commerce sites
        if db_param == 'bing_search':
            return self.PRODUCT_FOCUSED_PROMPT[0], self.PRODUCT_FOCUSED_PROMPT[1]
       
        # Check for custom prompts
        prompt_str, ans_struc = find_prompt(site, item_type, self.RANKING_PROMPT_NAME)
        if prompt_str is None:
            return self.RANKING_PROMPT[0], self.RANKING_PROMPT[1]
        else:
            return prompt_str, ans_struc
        
    def __init__(self, handler, items, ranking_type=FAST_TRACK, level="low"):
        ll = len(items)
        if ranking_type == self.FAST_TRACK:
            self.ranking_type_str = "FAST_TRACK"
        elif ranking_type == self.REGULAR_TRACK:
            self.ranking_type_str = "REGULAR_TRACK"
        elif ranking_type == self.WHO_RANKING:
            self.ranking_type_str = "WHO_RANKING"
        elif ranking_type == self.CONVERSATION_SEARCH:
            self.ranking_type_str = "CONVERSATION_SEARCH"
        else:
            self.ranking_type_str = "UNKNOWN"
        logger.info(f"Initializing Ranking with {ll} items, type: {self.ranking_type_str}")
        logger.info(f"Ranking {ll} items of type {self.ranking_type_str}")
        logger.info(f"Starting ranking of {ll} items (type: {self.ranking_type_str})")
        self.handler = handler
        self.level = level
        self.items = items
        self.num_results_sent = 0
        self.rankedAnswers = []
        self.ranking_type = ranking_type
#        self._results_lock = asyncio.Lock()  # Add lock for thread-safe operations

    async def rankItem(self, url, json_str, name, site):

        # Debug: Print site information
        # print(f"[rankItem] Processing item - URL: {url}, Site: {site}, Handler.site: {getattr(self.handler, 'site', 'NO SITE ATTR')}")

        if (self.ranking_type == Ranking.FAST_TRACK and self.handler.state.should_abort_fast_track()):
            logger.info("Fast track aborted, skipping item ranking")
            logger.info("Aborting fast track")
            return
        try:
            prompt_str, ans_struc = self.get_ranking_prompt()
            description = trim_json(json_str)
            prompt = fill_prompt(prompt_str, self.handler, {"item.description": description})
            ranking = await ask_llm(prompt, ans_struc, level=self.level, query_params=self.handler.query_params)
            
            # Handle both string and dictionary inputs for json_str
            schema_object = json_str if isinstance(json_str, dict) else json.loads(json_str)
            
            # If schema_object is an array, set it to the first item
            if isinstance(schema_object, list) and len(schema_object) > 0:
                schema_object = schema_object[0]
            
            ansr = {
                'url': url,
                'site': site,
                'name': name,
                'ranking': ranking,
                'schema_object': schema_object,
                'sent': False
            }

            record_llm_call(ansr, prompt_str, self.handler.query)
            
            # Check if required_item_type is specified and filter based on @type
            if self.handler.required_item_type is not None:
                item_type = schema_object.get('@type', None)
                if item_type != self.handler.required_item_type:
                    logger.debug(f"Item type mismatch: expected {self.handler.required_item_type}, got {item_type} - setting score to 0")
                    ranking["score"] = 0
            
            if (ranking["score"] > self.EARLY_SEND_THRESHOLD):
                logger.info(f"High score item: {name} (score: {ranking['score']}) - sending early {self.ranking_type_str}")
                try:
                    await self.sendAnswers([ansr])
                except (BrokenPipeError, ConnectionResetError):
                    logger.warning(f"Client disconnected while sending early answer for {name}")
                    self.handler.connection_alive_event.clear()
                    return
            
#            async with self._results_lock:  # Use lock when modifying shared state
            self.rankedAnswers.append(ansr)
            logger.debug(f"Item {name} added to ranked answers")
        
        except Exception as e:
            logger.error(f"Error in rankItem for {name}: {str(e)}")
            logger.debug(f"Full error trace: ", exc_info=True)
            # Import here to avoid circular import
            from config.config import CONFIG
            if CONFIG.should_raise_exceptions():
                raise  # Re-raise in testing/development mode

    def shouldSend(self, result):
        # Get max_results from handler, or use default
        max_results = getattr(self.handler, 'max_results', self.NUM_RESULTS_TO_SEND)

        # Don't send if we've already reached the limit
        if self.num_results_sent >= max_results:
            logger.debug(f"Not sending {result['name']} - already at limit ({self.num_results_sent}/{max_results})")
            return False

        should_send = False
        # Allow sending if we're still well below the limit
        if (self.num_results_sent < max_results - 3):
            should_send = True
        else:
            # Near the limit - only send if this result is better than something we already sent
            for r in self.rankedAnswers:
                if r["sent"] == True and r["ranking"]["score"] < result["ranking"]["score"]:
                    should_send = True
                    break

        logger.debug(f"Should send result {result['name']}? {should_send} (sent: {self.num_results_sent}/{max_results})")
        return should_send
    
    async def sendAnswers(self, answers, force=False):
        if not self.handler.connection_alive_event.is_set():
            logger.warning("Connection lost during ranking, skipping sending results")
            return
        
        # If this is FastTrack ranking, wait for prechecks to complete before sending
        if self.ranking_type == Ranking.FAST_TRACK:
            try:
                prechecks_done = await asyncio.wait_for(
                    self.handler.state.wait_for_prechecks(),
                    timeout=5.0
                )
                if not prechecks_done:
                    logger.info("Fast track aborted: prechecks not complete")
                    return
            except asyncio.TimeoutError:
                logger.warning("Fast track: prechecks timed out, not sending answers")
                return
                
            # Check abort conditions after prechecks
            if self.handler.state.should_abort_fast_track():
                logger.info("Fast track aborted, not sending answers")
                return
              
        json_results = []
        logger.debug(f"Considering sending {len(answers)} answers (force: {force})")

        # Get max_results from handler, or use default
        max_results = getattr(self.handler, 'max_results', self.NUM_RESULTS_TO_SEND)

        for result in answers:
            # Additional safety check - never exceed the limit even when forced
            if self.num_results_sent + len(json_results) >= max_results:
                logger.info(f"Stopping at {len(json_results)} results to avoid exceeding limit of {max_results}")
                break
                
            if self.shouldSend(result) or force:
                result_item = {
                    "@type": "Item",
                    "url": result["url"],
                    "name": result["name"],
                    "site": result["site"],
                    "siteUrl": result["site"],
                    "score": result["ranking"]["score"],
                    "description": result["ranking"]["description"],
                    "schema_object": result["schema_object"]
                }
                
                # Include query field for WHO ranking if present
                if self.ranking_type == Ranking.WHO_RANKING and "query" in result["ranking"]:
                    result_item["query"] = result["ranking"]["query"]
                
                json_results.append(result_item)
                
                result["sent"] = True
            
        if (json_results):  # Only attempt to send if there are results
            # Wait for pre checks to be done using event
            await self.handler.pre_checks_done_event.wait()
            
            # if we got here, prechecks are done. check once again for fast track abort
            if (self.ranking_type == Ranking.FAST_TRACK and self.handler.state.should_abort_fast_track()):
                logger.info("Fast track aborted after pre-checks")
                return
            
            try:
                # Final safety check before sending
                if self.num_results_sent + len(json_results) > max_results:
                    # Trim the results to not exceed the limit
                    allowed_count = max_results - self.num_results_sent
                    json_results = json_results[:allowed_count]
                    logger.warning(f"Trimmed results to {len(json_results)} to stay within limit of {max_results}")
                
                if (self.ranking_type == Ranking.FAST_TRACK):
                    self.handler.fastTrackWorked = True
                    logger.info("Fast track ranking successful")
                
                # Use the new schema to create and auto-send the message
                await create_assistant_result(json_results, handler=self.handler)
                self.num_results_sent += len(json_results)
                logger.info(f"Sent {len(json_results)} results, total sent: {self.num_results_sent}/{max_results}")
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Client disconnected while sending answers: {str(e)}")
                log(f"Client disconnected while sending answers: {str(e)}")
                self.handler.connection_alive_event.clear()
            except Exception as e:
                logger.error(f"Error sending answers: {str(e)}")
                log(f"Error sending answers: {str(e)}")
                self.handler.connection_alive_event.clear()
  
    async def sendMessageOnSitesBeingAsked(self, top_embeddings):
        if (self.handler.site == "all" or self.handler.site == "nlws"):
            sites_in_embeddings = {}
            for url, json_str, name, site in top_embeddings:
                sites_in_embeddings[site] = sites_in_embeddings.get(site, 0) + 1
            
            top_sites = sorted(sites_in_embeddings.items(), key=lambda x: x[1], reverse=True)[:3]
            top_sites_str = ", ".join([self.prettyPrintSite(x[0]) for x in top_sites])
            logger.info(f"Sending sites message: {top_sites_str}")
            
            try:
                # Create a custom message with asking_sites type
                message = Message(
                    sender_type=SenderType.SYSTEM,
                    message_type="asking_sites",  # Custom message type
                    content=top_sites_str,
                    conversation_id=self.handler.conversation_id if hasattr(self.handler, 'conversation_id') else None
                )
                asyncio.create_task(self.handler.send_message(message.to_dict()))
                self.handler.sites_in_embeddings_sent = True
            except (BrokenPipeError, ConnectionResetError):
                logger.warning("Client disconnected when sending sites message")
                self.handler.connection_alive_event.clear()
    
    async def do(self):
        logger.info(f"Starting ranking process with {len(self.items)} items")
        logger.info(f"Processing {len(self.items)} items for ranking")
        tasks = []
        for url, json_str, name, site in self.items:
            if self.handler.connection_alive_event.is_set():  # Only add new tasks if connection is still alive
                tasks.append(asyncio.create_task(self.rankItem(url, json_str, name, site)))
            else:
                logger.warning("Connection lost, not creating new ranking tasks")
       
        await self.sendMessageOnSitesBeingAsked(self.items)

        try:
            logger.debug(f"Running {len(tasks)} ranking tasks concurrently")
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during ranking tasks: {str(e)}")
            log(f"Error during ranking tasks: {str(e)}")

        if not self.handler.connection_alive_event.is_set():
            logger.warning("Connection lost during ranking, skipping sending results")
            log("Connection lost during ranking, skipping sending results")
            return

        # Wait for pre checks using event
        await self.handler.pre_checks_done_event.wait()
        
        if (self.ranking_type == Ranking.FAST_TRACK and self.handler.state.should_abort_fast_track()):
            logger.info("Fast track aborted after ranking tasks completed")
            return
    
        # Use min_score from handler if available, otherwise default to 51
        min_score_threshold = getattr(self.handler, 'min_score', 51)
        # Use max_results from handler if available, otherwise use NUM_RESULTS_TO_SEND
        max_results = getattr(self.handler, 'max_results', self.NUM_RESULTS_TO_SEND)
        filtered = [r for r in self.rankedAnswers if r['ranking']['score'] > min_score_threshold]
        ranked = sorted(filtered, key=lambda x: x['ranking']["score"], reverse=True)
        self.handler.final_ranked_answers = ranked[:max_results]

        logger.info(f"Filtered to {len(filtered)} results with score > {min_score_threshold}")
        logger.info(f"Filtered {len(self.rankedAnswers)} results to {len(filtered)} with score > {min_score_threshold}, returning top {len(self.handler.final_ranked_answers)}")
        logger.debug(f"Top 3 results: {[(r['name'], r['ranking']['score']) for r in ranked[:3]]}")

        results = [r for r in self.rankedAnswers if r['sent'] == False]
        if (self.num_results_sent > max_results):
            logger.info(f"Already sent {self.num_results_sent} results, returning without sending more")
            return
       
        # Sort by score in descending order
        sorted_results = sorted(results, key=lambda x: x['ranking']["score"], reverse=True)
        good_results = [x for x in sorted_results if x['ranking']["score"] > min_score_threshold]

        # Calculate how many more results we can send
        remaining_slots = max_results - self.num_results_sent
        if remaining_slots <= 0:
            logger.info(f"Already sent {self.num_results_sent} results, at or above limit of {max_results}")
            return
            
        if len(good_results) >= remaining_slots:
            tosend = good_results[:remaining_slots]
        else:
            tosend = good_results

        try:
            logger.info(f"Sending final batch of {len(tosend)} results")
            await self.sendAnswers(tosend, force=True)
        except (BrokenPipeError, ConnectionResetError):
            logger.error("Client disconnected during final answer sending")
            log("Client disconnected during final answer sending")
            self.handler.connection_alive_event.clear()

    def prettyPrintSite(self, site):
        ans = site.replace("_", " ")
        words = ans.split()
        return ' '.join(word.capitalize() for word in words)
