import requests

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
    }
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather
}