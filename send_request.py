import time
import json
import requests
from statistics import mean
import socket
import sys

SCRIPT_START_TIME = time.time()

def wait_for_port(port, host='0.0.0.0', timeout=300):
    """Wait for a port to be ready."""
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
        if time.time() - start_time > timeout:
            return False

def send_request(prompt, script_start_time, max_tokens=100, temperature=0.7, max_retries=5):
    url = "http://0.0.0.0:8000/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-2-13b-chat-hf",  # Adjust this to match your model
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    # Wait for the server to be ready
    print("Waiting for server to be ready...")
    if not wait_for_port(8000):
        print("Timeout waiting for server to be ready")
        return None

    for attempt in range(max_retries):
        try:
            request_start_time = time.time()
            first_token_time = None
            token_times = []
            full_response = ""

            with requests.post(url, headers=headers, json=data, stream=True, timeout=60) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            # Try to parse the line as JSON
                            chunk = json.loads(line.decode('utf-8'))
                        except json.JSONDecodeError:
                            # If it's not JSON, it might be SSE format
                            try:
                                chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                            except (IndexError, json.JSONDecodeError):
                                print(f"Couldn't parse line: {line.decode('utf-8')}")
                                continue

                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            token = chunk['choices'][0].get('text', '')
                            full_response += token
                            current_time = time.time()

                            if first_token_time is None:
                                first_token_time = current_time
                                time_to_first_token = first_token_time - script_start_time
                            else:
                                token_times.append(current_time - request_start_time)

            total_time = time.time() - request_start_time

            if len(token_times) > 1:
                avg_inter_token_latency = mean([token_times[i] - token_times[i-1] for i in range(1, len(token_times))])
            else:
                avg_inter_token_latency = 0

            return {
                "full_response": full_response,
                "time_to_first_token": time_to_first_token,
                "avg_inter_token_latency": avg_inter_token_latency,
                "total_time": total_time
            }

        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)  # Wait 5 seconds before retrying
            else:
                print("Max retries reached. Giving up.")
                return None

if __name__ == "__main__":
    prompt = "Once upon a time, in a land far away,"
    result = send_request(prompt, SCRIPT_START_TIME)
    if result:
        print(f"Full response: {result['full_response']}")
        print(f"Time to first token: {result['time_to_first_token']:.4f} seconds")
        print(f"Average inter-token latency: {result['avg_inter_token_latency']:.4f} seconds")
        print(f"Total time: {result['total_time']:.4f} seconds")
    else:
        print("Failed to get a response from the server.")
