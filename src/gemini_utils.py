"""
Gemini API integration for ArxivDigest.
This module provides functions to work with Google's Gemini API for paper analysis.
"""
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPIError
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiConfig:
    """Configuration for Gemini API calls."""
    def __init__(
        self,
        temperature: float = 0.4,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        use_thinking: bool = False
    ):
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.use_thinking = use_thinking

def setup_gemini_api(api_key: str) -> bool:
    """
    Setup the Gemini API with the provided API key.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    if not GEMINI_AVAILABLE:
        logger.error("Gemini package not installed. Run 'pip install google-generativeai'")
        return False
        
    if not api_key:
        logger.error("No Gemini API key provided")
        return False
        
    try:
        genai.configure(api_key=api_key)
        # Test API connection
        models = genai.list_models()
        logger.info(f"Successfully connected to Gemini API. Available models: {[m.name for m in models if 'generateContent' in m.supported_generation_methods]}")
        return True
    except Exception as e:
        logger.error(f"Failed to setup Gemini API: {e}")
        return False

def get_gemini_model(model_name: str = "gemini-2.5-flash", use_thinking: bool = False):
    """
    Get a Gemini model by name.
    
    Args:
        model_name: Name of the Gemini model
        use_thinking: Whether to enable thinking mode for supported models
        
    Returns:
        Model object or None if not available
    """
    if not GEMINI_AVAILABLE:
        return None
        
    try:
        # Configure system instruction for thinking mode if supported
        system_instruction = None
        if use_thinking and "2.5" in model_name:
            system_instruction = "Think step by step before responding. Show your reasoning process."
        
        # Configure safety settings to be minimal for academic content
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        if system_instruction:
            model = genai.GenerativeModel(
                model_name, 
                system_instruction=system_instruction,
                safety_settings=safety_settings
            )
        else:
            model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        return model
    except Exception as e:
        logger.error(f"Failed to get Gemini model: {e}")
        return None

def analyze_papers_with_gemini(
    papers: List[Dict[str, Any]], 
    query: Dict[str, str],
    config: Optional[GeminiConfig] = None,
    model_name: str = "gemini-2.5-flash"
) -> List[Dict[str, Any]]:
    """
    Analyze papers using the Gemini model.
    
    Args:
        papers: List of paper dictionaries
        query: Dictionary with 'interest' key describing research interests
        config: GeminiConfig object
        model_name: Name of the Gemini model to use
        
    Returns:
        List of papers with added analysis
    """
    if not GEMINI_AVAILABLE:
        logger.error("Gemini package not installed. Cannot analyze papers.")
        return papers
        
    if not config:
        config = GeminiConfig()
        
    model = get_gemini_model(model_name, config.use_thinking if config else False)
    if not model:
        return papers
        
    analyzed_papers = []
    
    for paper in papers:
        try:
            # Prepare prompt
            prompt = f"""
            You are a research assistant analyzing academic papers in AI and ML.
            
            Analyze this paper and provide insights based on the user's research interests.
            
            Research interests: {query['interest']}
            
            Paper details:
            Title: {paper['title']}
            Authors: {paper['authors']}
            Abstract: {paper['abstract']}
            Content: {paper['content'][:5000]}
            
            Please provide your response as a single JSON object with the following structure:
            {{
              "Relevancy score": 1-10 (higher = more relevant),
              "Reasons for match": "Detailed explanation of why this paper matches the interests",
              "Key innovations": "List the main contributions of the paper",
              "Critical analysis": "Evaluate strengths and weaknesses",
              "Goal": "What problem does the paper address?",
              "Data": "Description of datasets used",
              "Methodology": "Technical approach and methods",
              "Implementation details": "Model architecture, hyperparameters, etc.",
              "Experiments & Results": "Key findings and comparisons",
              "Discussion & Next steps": "Limitations and future work",
              "Related work": "Connection to similar research",
              "Practical applications": "Real-world uses of this research",
              "Key takeaways": ["Point 1", "Point 2", "Point 3"]
            }}
            
            Format your response as a valid JSON object and nothing else.
            """
            
            # Just log that we're sending a prompt to Gemini
            print(f"Sending prompt to Gemini for paper: {paper['title'][:50]}...")
            
            generation_config = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "max_output_tokens": config.max_output_tokens,
            }
            
            # Disable safety settings for academic content analysis
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check if response was blocked or has no content
            if not response.candidates:
                logger.warning(f"No candidates returned for paper {paper['title']} - response may have been blocked")
                paper['gemini_analysis'] = {"error": "Response blocked or no candidates returned"}
                analyzed_papers.append(paper)
                continue
                
            candidate = response.candidates[0]
            
            # Check finish reason
            if candidate.finish_reason == 2:  # SAFETY
                logger.warning(f"Response blocked by safety filters for paper {paper['title']}")
                paper['gemini_analysis'] = {"error": "Response blocked by safety filters"}
                analyzed_papers.append(paper)
                continue
            elif candidate.finish_reason == 3:  # RECITATION
                logger.warning(f"Response blocked due to recitation for paper {paper['title']}")
                paper['gemini_analysis'] = {"error": "Response blocked due to recitation"}
                analyzed_papers.append(paper)
                continue
            elif candidate.finish_reason not in [0, 1]:  # 0=UNSPECIFIED, 1=STOP (normal completion)
                logger.warning(f"Unexpected finish reason {candidate.finish_reason} for paper {paper['title']}")
                paper['gemini_analysis'] = {"error": f"Unexpected finish reason: {candidate.finish_reason}"}
                analyzed_papers.append(paper)
                continue
            
            # Check if content exists
            if not candidate.content or not candidate.content.parts:
                logger.warning(f"No content in response for paper {paper['title']}")
                paper['gemini_analysis'] = {"error": "No content in response"}
                analyzed_papers.append(paper)
                continue
            
            # Extract and parse the response
            try:
                response_text = response.text
            except Exception as e:
                logger.warning(f"Failed to extract text from response for paper {paper['title']}: {e}")
                paper['gemini_analysis'] = {"error": f"Failed to extract text: {str(e)}"}
                analyzed_papers.append(paper)
                continue
            
            # Try to extract JSON
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    try:
                        gemini_analysis = json.loads(json_str)
                        
                        # Add Gemini analysis to paper
                        paper['gemini_analysis'] = gemini_analysis
                        
                        # Directly copy fields to paper
                        for key, value in gemini_analysis.items():
                            paper[key] = value
                            
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"Failed to parse JSON for paper {paper['title']}: {json_err}")
                        # Try to create a minimal fallback analysis
                        fallback_analysis = create_fallback_analysis(paper, response_text)
                        paper['gemini_analysis'] = fallback_analysis
                        for key, value in fallback_analysis.items():
                            paper[key] = value
                else:
                    logger.warning(f"Could not extract JSON from Gemini response for paper {paper['title']}")
                    # Create fallback analysis when no JSON structure found
                    fallback_analysis = create_fallback_analysis(paper, response_text)
                    paper['gemini_analysis'] = fallback_analysis
                    for key, value in fallback_analysis.items():
                        paper[key] = value
            except Exception as e:
                logger.warning(f"Error processing response for paper {paper['title']}: {e}")
                # Create minimal fallback analysis
                fallback_analysis = create_fallback_analysis(paper, "Error processing response")
                paper['gemini_analysis'] = fallback_analysis
                for key, value in fallback_analysis.items():
                    paper[key] = value
                
            analyzed_papers.append(paper)
            
            # Avoid rate limiting
            time.sleep(1)
            
        except GoogleAPIError as e:
            logger.error(f"Gemini API error: {e}")
            paper['gemini_analysis'] = {"error": f"Gemini API error: {str(e)}"}
            analyzed_papers.append(paper)
            
        except Exception as e:
            logger.error(f"Error analyzing paper with Gemini: {e}")
            paper['gemini_analysis'] = {"error": f"Error: {str(e)}"}
            analyzed_papers.append(paper)
            
    return analyzed_papers

def get_topic_clustering(papers: List[Dict[str, Any]], model_name: str = "gemini-2.5-flash", use_thinking: bool = False):
    """
    Cluster papers by topic using Gemini.
    
    Args:
        papers: List of paper dictionaries
        model_name: Name of the Gemini model to use
        
    Returns:
        Dictionary with topic clusters
    """
    if not GEMINI_AVAILABLE:
        logger.error("Gemini package not installed. Cannot cluster papers.")
        return {}
        
    model = get_gemini_model(model_name, use_thinking)
    if not model:
        return {}
        
    # Create a condensed representation of the papers
    paper_summaries = []
    for i, paper in enumerate(papers):
        paper_summaries.append(f"{i+1}. Title: {paper['title']}\nAbstract: {paper['abstract'][:300]}...")
    
    paper_text = "\n\n".join(paper_summaries)
    
    prompt = f"""
    You are a research librarian organizing academic papers into topic clusters.
    
    Analyze these papers and group them into 3-7 thematic clusters:
    
    {paper_text}
    
    For each cluster:
    1. Provide a descriptive name for the cluster
    2. List the paper numbers that belong to this cluster
    3. Explain why these papers belong together
    
    Format your response as JSON with these fields: "clusters" (an array of objects with "name", "papers", and "description" fields).
    """
    
    try:
        # Disable safety settings for clustering academic content
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        # Check if response was blocked
        if not response.candidates or not response.candidates[0].content:
            logger.warning("Clustering response was blocked or has no content")
            return {"error": "Response blocked or no content"}
            
        response_text = response.text
        
        # Try to extract JSON
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                cluster_data = json.loads(json_str)
                return cluster_data
            else:
                logger.warning("Could not extract JSON from Gemini clustering response")
                return {"error": "Failed to parse clustering response"}
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini clustering response as JSON")
            return {"error": "Failed to parse clustering response"}
            
    except Exception as e:
        logger.error(f"Error clustering papers with Gemini: {e}")
        return {"error": f"Clustering error: {str(e)}"}

def create_fallback_analysis(paper: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    """
    Create a fallback analysis when JSON parsing fails.
    
    Args:
        paper: Paper dictionary
        response_text: Raw response text from Gemini
        
    Returns:
        Dictionary with fallback analysis
    """
    # Extract any numeric score from the response text
    import re
    scores = re.findall(r'\b([1-9]|10)\b', response_text)
    relevancy_score = int(scores[0]) if scores else 5  # Default to middle score
    
    # Try to extract key phrases
    reasons = "Analysis could not be fully parsed from response"
    if len(response_text) > 50:
        # Take first 200 characters as reasons
        reasons = response_text[:200].strip()
        if not reasons.endswith('.'):
            reasons += "..."
    
    return {
        "Relevancy score": relevancy_score,
        "Reasons for match": reasons,
        "Key innovations": "Could not parse from response",
        "Critical analysis": "Analysis parsing failed",
        "Goal": "Could not determine from response",
        "Data": "Not specified",
        "Methodology": "Could not parse methodology",
        "Implementation details": "Not available",
        "Experiments & Results": "Could not parse results",
        "Discussion & Next steps": "Not available",
        "Related work": "Not specified",
        "Practical applications": "Could not determine",
        "Key takeaways": ["Analysis parsing failed", "Please check response manually"],
        "parsing_error": True,
        "raw_response": response_text[:500]  # Store first 500 chars for debugging
    }

def analyze_with_gemini_2_5_flash_thinking(
    papers: List[Dict[str, Any]], 
    query: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Analyze papers using Gemini 2.5 Flash with thinking enabled.
    
    Args:
        papers: List of paper dictionaries
        query: Dictionary with 'interest' key describing research interests
        
    Returns:
        List of papers with added analysis
    """
    config = GeminiConfig(use_thinking=True)
    return analyze_papers_with_gemini(papers, query, config, "gemini-2.5-flash")

def analyze_with_gemini_2_5_flash_no_thinking(
    papers: List[Dict[str, Any]], 
    query: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Analyze papers using Gemini 2.5 Flash without thinking.
    
    Args:
        papers: List of paper dictionaries
        query: Dictionary with 'interest' key describing research interests
        
    Returns:
        List of papers with added analysis
    """
    config = GeminiConfig(use_thinking=False)
    return analyze_papers_with_gemini(papers, query, config, "gemini-2.5-flash")

def analyze_with_gemini_2_5_pro(
    papers: List[Dict[str, Any]], 
    query: Dict[str, str],
    use_thinking: bool = False
) -> List[Dict[str, Any]]:
    """
    Analyze papers using Gemini 2.5 Pro.
    
    Args:
        papers: List of paper dictionaries
        query: Dictionary with 'interest' key describing research interests
        use_thinking: Whether to enable thinking mode
        
    Returns:
        List of papers with added analysis
    """
    config = GeminiConfig(
        use_thinking=use_thinking,
        max_output_tokens=4096,  # Higher token limit for Pro model
        temperature=0.3  # Slightly lower temperature for more consistency
    )
    return analyze_papers_with_gemini(papers, query, config, "gemini-2.5-pro")