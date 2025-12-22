import requests
import json
import time
import os
import sys

URL = "http://127.0.0.1:5000/api/transcribe_long"

# Use command line argument for file if provided, otherwise default to user request
if len(sys.argv) > 1:
    FILE_PATH = sys.argv[1]
else:
    FILE_PATH = "anis.wav" 

def test_long_transcription():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Test file {FILE_PATH} not found.")
        return

    print(f"Sending {FILE_PATH} to {URL}...")
    
    try:
        with open(FILE_PATH, 'rb') as f:
            files = {'file': f}
            data = {'lang_code': 'ben_Beng'}
            start_time = time.time()
            response = requests.post(URL, files=files, data=data, timeout=600)
            end_time = time.time()
            
        print(f"Request took {end_time - start_time:.2f} seconds.")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✓ Success!")
                segments = result.get("results", [])
                print(f"Received {len(segments)} segments.")
                for seg in segments:
                    print(f"  Segment {seg['segment']}: {seg['text'][:50]}...")
            else:
                 print(f"✗ Server returned success=False: {result}")
        else:
            print(f"✗ Failed with status {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"✗ Exception: {e}")

if __name__ == "__main__":
    test_long_transcription()
