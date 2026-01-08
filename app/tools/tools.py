import requests
from tavily import TavilyClient
from langchain_core.tools import tool
import os

from app.data.data import supports_tools

def is_tool_supported(model_name):
    """Check if the given model supports tool usage.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model supports tools, False otherwise.
    """
    return model_name in supports_tools

@tool
def weather_tool(location: str) -> dict:
    """Get the current weather for a given city.
    
    Args:
        location: The name of the city to get weather for (e.g., "Helsinki", "New York").
    
    Returns:
        Weather information including temperature, description, humidity, and wind speed.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    return get_weather(location, api_key)

@tool
def search_tool(query: str, max_results: int = 5) -> dict:
    """Search the web for current information on any topic.
    
    Use this when you need up-to-date information, facts, news, or answers to questions
    that require current data beyond your knowledge cutoff.

    Args:
        query: The search query. Be specific and clear about what information you're looking for.
        max_results: Maximum number of search results to return (default: 5, max: 10).
    
    Returns:
        Search results containing titles, URLs, content snippets, and relevance scores.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    return web_search(query, api_key, max_results)

def get_weather(city_name: str, api_key):
    """Fetch weather data for a given city.

    Args:
    city_name (str): Name of the city (e.g., "Helsinki").
    api_key (str): Your OpenWeather API key.

    Returns:
      dict: Weather information including temperature, description, etc., or error message.
    """

    if not api_key:
        return {"error": "Missing API key."}

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params={
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(data)
        if data.get('cod') != 200:
            return {"error": data.get("message", "Unknown error")}
        
        weather = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data['main']['temp'],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather
    except requests.RequestException as e:
        return {"error": str(e)}
    
def web_search(query, api_key, max_results=5):
    """Perform a web search using Tavily API.

    Args:
        query (str): The search query string.
        api_key (str): Your Tavily API key.
        max_results (int): Maximum number of search results to return (default: 5).

    Returns:
        dict: Search results containing titles, URLs, and content snippets, or error message.
    """
    
    if not api_key:
        return {"error": "Missing Tavily API key."}
    
    try:
        # Initialize the Tavily client with your API key
        client = TavilyClient(api_key=api_key)
        
        # Perform the search
        # search_depth can be "basic" or "advanced"
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )
        
        # Extract and format the results
        results = []
        for result in response.get('results', []):
            results.append({
                "title": result.get("title", "No title"),
                "url": result.get("url", ""),
                "content": result.get("content", "No content available"),
                "score": result.get("score", 0)  # Relevance score
            })
        
        return {
            "query": query,
            "results": results,
            "answer": response.get("answer", "")  # Tavily sometimes provides a direct answer
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

LANGCHAIN_TOOLS = [weather_tool, search_tool]