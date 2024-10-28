import requests

# Define the coupon receiver data
coupon_receiver = {
    "destination": ["No Urgent Place", "Home", "Work"],
    "weather": ["Sunny", "Rainy", "Snowy"],
    "time": ["10AM", "10PM", "7AM"],
    "coupon": ["Coffee House", "Coffee House", "Coffee House"],
    "expiration": ["2h", "2h", "1d"],
    "same_direction": [0, 1, 1],
    "coupon_accepting": [0, 0, 0],
}

# Define the URL of your web service
URL = "http://localhost:9696/predict"

try:
    # Send a POST request to the web service
    response = requests.post(URL, json=coupon_receiver, timeout=5)
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Print the JSON response
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")