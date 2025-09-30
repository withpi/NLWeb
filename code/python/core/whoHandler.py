from core.baseHandler import NLWebHandler
from core.retriever import search
from core.whoRanking import WhoRanking
from misc.logger.logging_config_helper import get_configured_logger

# Who handler is work in progress for answering questions about who
# might be able to answer a given query

logger = get_configured_logger("who_handler")

DEFAULT_NLWEB_ENDPOINT = "https://nlwm.azurewebsites.net/ask"

class WhoHandler (NLWebHandler) :

    def __init__(self, query_params, http_handler): 
        # Remove site parameter - we'll use nlweb_sites
        if 'site' in query_params:
            del query_params['site']
            
        # Keep prev_queries if provided for context, but don't use 'prev' format
        # The who handler can use previous queries to understand follow-up questions
        if 'prev' in query_params:
            del query_params['prev']
        # Keep prev_queries for context if provided
        super().__init__(query_params, http_handler)
    
    def _build_nlweb_url(self, site_url, site_type=None):
        """Helper function to build the complete NLWEB URL with all parameters."""
        from urllib.parse import quote

        params = []
        params.append(f"site={site_url}")

        # Add the user's query
        if self.query:
            params.append(f"query={quote(self.query)}")

        # Check if it's a Shopify site and add db parameter
        if site_type in ['ShopifyStore', 'Shopify'] or 'shopify' in site_url.lower():
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
            if 'content' in message and isinstance(message['content'], list):
                for result in message['content']:
                    if 'url' in result:
                        url = result['url']
                        # If URL doesn't start with http:// or https://, convert to /ask endpoint
                        if not url.startswith(('http://', 'https://')):
                            site_type = result.get('@type', '')
                            result['url'] = self._build_nlweb_url(url, site_type)
                            logger.debug(f"Modified URL from '{url}' to '{result['url']}'")

            # Handle single result messages
            elif 'url' in message:
                url = message['url']
                if not url.startswith(('http://', 'https://')):
                    site_type = message.get('@type', '')
                    message['url'] = self._build_nlweb_url(url, site_type)
                    logger.debug(f"Modified URL from '{url}' to '{message['url']}'")

        # Call parent class's send_message with modified message
        await super().send_message(message)

    async def runQuery(self):

        try:
            # Always use general search with nlweb_sites
            logger.info("Using general search method with site=nlweb_sites for who query")
            
            # Search using the special nlweb_sites collection
            items = await search(
                self.query, 
                site='nlweb_sites',  # Use the sites collection
                query_params=self.query_params,
                num_results=25
            )
            self.final_retrieved_items = items
            print(f"\n=== WHO HANDLER: Retrieved {len(items)} items from nlweb_sites ===")
            
            # Print just the site names
            print("\nRetrieved sites:")
            site_names = []
            for item in items:
                if isinstance(item, tuple) and len(item) >= 3:
                    name = item[2]  # name is the third element
                    site_names.append(name)
            
            # Print unique site names
            unique_sites = sorted(set(site_names))
            for i, name in enumerate(unique_sites, 1):
                print(f"  {i}. {name}")
            print("=" * 60)
            
            logger.debug(f"Who retrieval complete: {len(self.final_retrieved_items)} items retrieved")
            
            # Use simplified WHO ranking - no decontextualization needed
            self.ranker = WhoRanking(self, self.final_retrieved_items)
            await self.ranker.do()
            
            logger.info("Who ranking completed")
            logger.debug("Who ranking complete")
            return self.return_value  
                
        except Exception as e:
            raise
        