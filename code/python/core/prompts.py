# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file is used to get the right prompt for a given type, site and prompt-name.
Also deals with filling in the prompt and running prompts.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from xml.etree import ElementTree as ET
import json 
import os  # Add this import
from misc.logger.logging_config_helper import get_configured_logger
from core.llm import ask_llm
from core.config import CONFIG

logger = get_configured_logger("prompts")
prompt_runner_logger = get_configured_logger("prompt_runner")


BASE_NS = "http://nlweb.ai/base"
SITE_TAG = "{" + BASE_NS + "}Site"
PROMPT_TAG = "{" + BASE_NS + "}Prompt"
PROMPT_STRING_TAG = "{" + BASE_NS + "}promptString"
RETURN_STRUC_TAG = "{" + BASE_NS + "}returnStruc"

# This file deals with getting the right prompt for a given
# type, site and prompt-name. 
# Also deals with filling in the prompt.
# #Yet to do the subclass check.

prompt_roots = []
def init_prompts(files=["prompts.xml"]):
    global prompt_roots
    logger.info(f"Initializing prompts from files: {files}")
    
    for file in files:
        # Create full path by joining the config directory with the filename
        file_path = os.path.join(CONFIG.config_directory, file)
        try:
            prompt_roots.append(ET.parse(file_path).getroot())
        except Exception as e:
            logger.error(f"Failed to load prompt file '{file}': {str(e)}")
            raise


def super_class_of(child_class, parent_class):
    if parent_class == child_class:
        return True
    if parent_class == "{" + BASE_NS + "}Item" :
        return True
    return False

prompt_var_cache = {}
def get_prompt_variables_from_prompt(prompt):
    if prompt in prompt_var_cache:
        return prompt_var_cache[prompt]
    variables = extract_variables_from_prompt(prompt)
    prompt_var_cache[prompt] = variables
    return variables

def extract_variables_from_prompt(prompt):
    # Find all strings between { and }
    variables = set()
    start = 0
    while True:
        # Find next opening brace
        start = prompt.find('{', start)
        if start == -1:
            break
            
        # Find matching closing brace
        end = prompt.find('}', start)
        if end == -1:
            break
            
        # Extract variable name and add to set
        var = prompt[start+1:end].strip()
        variables.add(var)
        
        # Move start position
        start = end + 1
    return variables

def get_prompt_variable_value(variable, handler):
    
    site = handler.site
    query = handler.query
    prev_queries = handler.prev_queries
    value = ""

    if variable == "request.site":
        if (isinstance(site, list)):
            value = site
    elif variable == "site.itemType":
        item_type = handler.item_type
        value = item_type.split("}")[1]
    elif variable == "request.query":
        if (handler.state.is_decontextualization_done()):
            value = handler.decontextualized_query
        elif (len(prev_queries) > 0):
            value = query + " previous queries: " + str(prev_queries)
        else:
            value = query
    elif variable == "request.previousQueries":
        value = str(prev_queries)
    elif variable == "request.contextUrl":
        value = handler.context_url
    elif variable == "request.itemType":
        value = handler.item_type
    elif variable == "request.contextDescription":
        value = handler.context_description
    elif variable == "request.rawQuery":
        value = query
    elif variable == "request.prevAnswers":
        # Get previous answers from handler - the attribute is named 'last_answers'
        last_answers = getattr(handler, 'last_answers', [])
        if last_answers:
            value = str(last_answers)
        else:
            value = ""
    elif variable == "request.answers":
        value = str(handler.final_ranked_answers)
    elif variable == "tool.description":
        value = getattr(handler.tool, 'description', '')
    elif variable == "tools.description":
        value = getattr(handler.tools, 'description', '')
    elif variable == "request.top_k":
        value = str(getattr(handler, 'top_k', 3))
    elif variable == "request.item_name":
        value = getattr(handler, 'item_name', '')
    elif variable == "request.details_requested":
        value = getattr(handler, 'details_requested', '')
    else:
        logger.warning(f"Unknown variable: {variable}")
        value = ""
    
    return value

def fill_prompt(prompt_str, handler, pr_dict={}):
    try:
        variables = get_prompt_variables_from_prompt(prompt_str)
        for variable in variables:
            if (variable in pr_dict):
                value = pr_dict[variable]
            else:
                value = get_prompt_variable_value(variable, handler)
            # Ensure value is a string
            if not isinstance(value, str):
                value = str(value)
                
            prompt_str = prompt_str.replace("{" + variable + "}", value)
        return prompt_str
    except Exception as e:
        logger.error(f"Error filling prompt: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        raise


cached_prompts = {}
def get_cached_values(site, item_type, prompt_name):
    cache_key = (site, item_type, prompt_name)
    if cache_key in cached_prompts:
        return cached_prompts[cache_key]
    return None

def find_prompt(site, item_type, prompt_name):  
    if site and isinstance(site, list):
        site = site[0]
    if (prompt_roots == []):
        init_prompts()
    
    cached_values = get_cached_values(site, item_type, prompt_name)
    if cached_values is not None:
        return cached_values
    
    # First, try to find a Site element matching the site parameter
    site_element = None
    prompt_element = None
    default_site_element = None
    default_item_type_element = []
    item_type_element = []
    prompt_element = None

    for root_element in prompt_roots:
        for se in root_element.findall(SITE_TAG):
            se_site = se.get("id")
            if se_site == site :
                site_element = se
                break
            if se_site == "default":
                default_site_element = se

    if (not site_element and default_site_element):
        site_element = default_site_element
    
    if (not site_element):
        return None, None

  
    for child in site_element:
        if child.tag == "{" + BASE_NS + "}Item":
            default_item_type_element.append(child)
        elif child.tag == "{" + BASE_NS + "}item_type":
            item_type_element.append(child)
            
    for elt in default_item_type_element:
        item_type_element.append(elt)
       
    for elt in item_type_element:
        prompts = elt.findall(PROMPT_TAG)
        # Debug: prompts found for site and item_type
        for pe in prompts:
            # Debug: prompt reference
            if pe.get("ref") == prompt_name:
                prompt_element = pe
                break
            if (prompt_element):
                break
            
        
    if (not prompt_element):
        # Debug: prompt not found
        return None, None
    else:
        prompt_text = prompt_element.find(PROMPT_STRING_TAG).text
        return_struc_element = prompt_element.find(RETURN_STRUC_TAG)
        
        if return_struc_element is not None and return_struc_element.text:
            return_struc_text = return_struc_element.text.strip()
            if return_struc_text == "":
                return_struc = None
            else:
                try:
                    return_struc = json.loads(return_struc_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse return structure JSON: {e}")
                    return_struc = None
        else:
            return_struc = None
        
        cached_prompts[(site, item_type, prompt_name)] = (prompt_text, return_struc)
        return prompt_text, return_struc
    


def get_prompt_variables_from_file(xml_file_path):
    """
    Parse XML file and extract variables from promptString elements.
    Returns a set of all variables found.
    """
    logger.info(f"Extracting prompt variables from file: {xml_file_path}")
    
    try:
        # Parse XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        logger.debug(f"Successfully parsed XML file: {xml_file_path}")
        
        # Find all promptString elements recursively
        all_variables = set()
        
        def process_element(element):
            # Check if current element is a promptString
            if element.tag == PROMPT_STRING_TAG:
                prompt_text = element.text
                if prompt_text:
                    variables = extract_variables_from_prompt(prompt_text)
                    all_variables.update(variables)
                    logger.debug(f"Found {len(variables)} variables in promptString")
            
            # Recursively process all child elements
            for child in element:
                process_element(child)
                
        # Start recursive processing from root
        process_element(root)
        
        logger.info(f"Extracted {len(all_variables)} unique variables from {xml_file_path}")
        logger.debug(f"Variables found: {all_variables}")
        return all_variables
        
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file {xml_file_path}: {str(e)}")
        return set()
    except FileNotFoundError:
        logger.error(f"XML file not found: {xml_file_path}")
        return set()
    except Exception as e:
        logger.error(f"Error processing file {xml_file_path}: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        return set()

#print(get_prompt_variables_from_file("html/site_type.xml"))


class PromptRunner:
    """Class to run prompts with a given handler."""

    def get_prompt(self, prompt_name):
        item_type = self.handler.item_type
        site = self.handler.site
        
        # Hardcoded PrevQueryDecontextualizer prompt
        if prompt_name == 'PrevQueryDecontextualizer':
            prompt_str = """The user is querying the site {request.site} which has {site.itemType}s.
        Rewrite the query, incorporating the context of the previous queries and answers.
        Keep the decontextualized query short and do not reference the site. 

        If the query very clearly does not reference earlier queries, 
        don't change the query. Err on the side of incorporating the context of the 
        previous queries. If you are not sure whether this is a brand new query, 
        or follow up, it is likely a follow up. Try your best to incorporate the 
        context from the previous queries.

        The user's query is: {request.rawQuery}. 
        Previous queries were: {request.previousQueries}."""
            
            ans_struc = {
                "requires_decontextualization": "True or False",
                "decontextualized_query": "The rewritten query, if decontextualization is required"
            }
            return prompt_str, ans_struc
        
        # For other decontextualization prompts, use 'default' site to find root-level prompts
        if 'Decontextualizer' in prompt_name:
            site = 'default'
        
        prompt_str, ans_struc = find_prompt(site, item_type, prompt_name)

        if (prompt_str is None):
            prompt_runner_logger.warning(f"Prompt '{prompt_name}' not found for site='{site}', item_type='{item_type}'")
            return None, None
        
        return prompt_str, ans_struc

    def __init__(self, handler):
        self.handler = handler

    async def run_prompt(self, prompt_name, level="low", verbose=False, timeout=8):
        prompt_runner_logger.info(f"Running prompt: {prompt_name} with level={level}, timeout={timeout}s")
        
        try:
            prompt_str, ans_struc = self.get_prompt(prompt_name)
            if (prompt_str is None):
                if (verbose):
                    print(f"Prompt {prompt_name} not found")
                prompt_runner_logger.debug(f"Cannot run prompt '{prompt_name}' - prompt not found")
                return None
        
            prompt_runner_logger.debug(f"Filling prompt template with handler data")
            prompt = fill_prompt(prompt_str, self.handler)
            if (verbose):
                print(f"Prompt: {prompt}")
            prompt_runner_logger.debug(f"Filled prompt length: {len(prompt)} chars")
            
            prompt_runner_logger.info(f"Calling LLM with level={level}")
            response = await ask_llm(prompt, ans_struc, level=level, timeout=timeout, query_params=self.handler.query_params)
            
            if response is None:
                prompt_runner_logger.warning(f"LLM returned None for prompt '{prompt_name}'")
            else:
                prompt_runner_logger.info(f"LLM response received for prompt '{prompt_name}'")
                prompt_runner_logger.debug(f"Response type: {type(response)}, size: {len(str(response))} chars")
            
            if (verbose):
                print(f"Response: {response}")
            
            return response
            
        except Exception as e:
            from core.config import CONFIG
            error_msg = f"Error in run_prompt for '{prompt_name}': {type(e).__name__}: {str(e)}"
            prompt_runner_logger.error(error_msg)
            prompt_runner_logger.debug("Full traceback:", exc_info=True)
            
            if CONFIG.should_raise_exceptions():
                # In testing/development mode, re-raise with enhanced error message
                raise Exception(f"LLM call failed for prompt '{prompt_name}': {type(e).__name__}: {str(e)}") from e
            else:
                # In production mode, log and return None
                logger.error(f"ERROR in run_prompt: {type(e).__name__}: {str(e)}")
                return None
