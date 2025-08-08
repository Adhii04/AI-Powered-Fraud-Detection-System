import requests
import json
import time

def test_api():
    url = "http://127.0.0.1:8001/predict"
    data = {
        "trans_date_trans_time": "2024-08-08 10:00:00",
        "cc_num": "4532789123456789",
        "merchant": "Amazon",
        "category": "shopping",
        "amt": 299.99,
        "first": "John",
        "last": "Doe",
        "gender": "M",
        "street": "123 Main St",
        "city": "New York",
        "state": "NY",
        "zip": "10001",
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8419000,
        "job": "Software Engineer",
        "dob": "1990-01-01",
        "trans_num": "TR123456789",
        "unix_time": 1691481600,
        "merch_lat": 47.6062,
        "merch_long": -122.3321,
        "day_of_week": "Thursday",
        "hour_of_day": 10,
        "age": 34,
        "distance_km": 3862.5
    }

    print("Waiting for server to start...")
    time.sleep(2)
    
    try:
        print("Testing connection to server...")
        response = requests.get("http://127.0.0.1:8001")
        print(f"Server response (GET /): {response.json()}")
        
        print("\nTesting prediction endpoint...")
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_api()
