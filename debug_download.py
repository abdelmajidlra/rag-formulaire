import requests

url = "https://www.canada.ca/content/dam/ircc/migration/ircc/francais/pdf/trousses/form/imm5707f.pdf"

print(f"Attempting to download: {url}")

try:
    # Try without headers (like current script)
    print("\n--- Request without headers ---")
    resp = requests.get(url, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Headers: {resp.headers}")
    print(f"First 100 bytes: {resp.content[:100]}")
    
    if resp.content.startswith(b'%PDF-'):
        print("Result: VALID PDF header")
    else:
        print("Result: INVALID header")

    # Try with User-Agent
    print("\n--- Request with User-Agent ---")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"First 100 bytes: {resp.content[:100]}")
    
    if resp.content.startswith(b'%PDF-'):
        print("Result: VALID PDF header")
    else:
        print("Result: INVALID header")

except Exception as e:
    print(f"Error: {e}")
