import requests
from tavily import TavilyClient
from data import supports_tools

def is_tool_supported(model_name):
    """Check if the given model supports tool usage.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model supports tools, False otherwise.
    """
    return model_name in supports_tools

def get_weather(city_name, api_key):
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

ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city to get weather for."
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic. Use this when you need up-to-date information, facts, news, or answers to questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and clear about what information you're looking for."
                    }
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of search results to return (default is 5, max: 10).",
                    "default": 5
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "web_search": web_search
}