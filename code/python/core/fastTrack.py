# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file contains the code for the 'fast track' path, which assumes that the query is a simple question,
not requiring decontextualization, query is relevant, the query has all the information needed, etc.
Those checks are done in parallel with fast track. Results are sent to the client only after
all those checks are done, which should arrive by the time the results are ready.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from core.retriever import search
import core.ranking as ranking
from misc.logger.logging_config_helper import get_configured_logger
from core.config import CONFIG
import asyncio

logger = get_configured_logger("fast_track")

# Sites that don't support standard vector retrieval
NO_STANDARD_RETRIEVAL_SITES = ["datacommons", "all", "conv_history", "CricketLens", "cricketlens", "cricketlens.com"]

def site_supports_standard_retrieval(site):
    """Check if a site supports standard vector database retrieval"""
    
    # If site is "all" and aggregation is disabled, treat it as supporting standard retrieval
    if site == "all" and not CONFIG.is_aggregation_enabled():
        logger.debug("Site is 'all' with aggregation disabled - treating as standard retrieval")
        return True
    
    return site not in NO_STANDARD_RETRIEVAL_SITES

class FastTrack:
    def __init__(self, handler):
        self.handler = handler
        logger.debug("FastTrack initialized")

    def is_fastTrack_eligible(self):
        """Check if query is eligible for fast track processing"""
        # Skip fast track for sites without standard retrieval
        return False
        if not site_supports_standard_retrieval(self.handler.site):
            return False
        if (self.handler.context_url != ''):
            logger.debug("Fast track not eligible: context_url present")
            return False
        if (len(self.handler.prev_queries) > 0):
            logger.debug(f"Fast track not eligible: {len(self.handler.prev_queries)} previous queries present")
            return False
        logger.info("Query is eligible for fast track")
        return True
        
    async def do(self):
        """Execute fast track processing"""
        if (not self.is_fastTrack_eligible()):
            logger.info("Fast track processing skipped - not eligible")
            return
        
        logger.info("Starting fast track processing")
        print(f"using fast track")
        self.handler.retrieval_done_event.set()  # Use event instead of flag
        
        try:
           
            items = await search(
                self.handler.query, 
                self.handler.site,
                query_params=self.handler.query_params,
                handler=self.handler
            )
            self.handler.final_retrieved_items = items
          
            if (not self.handler.query_done and not self.handler.abort_fast_track_event.is_set()):
                self.handler.fastTrackRanker = ranking.Ranking(self.handler, items, ranking.Ranking.FAST_TRACK)
                await self.handler.fastTrackRanker.do()
                return
                
        except Exception as e:
            logger.error(f"Error during fast track processing: {str(e)}")
            raise