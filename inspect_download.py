import requests
import sys

url = "https://www.canada.ca/content/dam/ircc/migration/ircc/francais/pdf/trousses/form/imm5707f.pdf"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

print(f"Downloading {url}...")
try:
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Headers: {resp.headers}")
    
    content = resp.content
    print(f"Length: {len(content)}")
    print(f"Start: {content[:200]}")
    
    if b'<!doctype' in content.lower() or b'<html' in content.lower():
        print("\n--- HTML CONTENT DETECTED ---")
        print(content.decode('utf-8', errors='ignore')[:1000])
        
except Exception as e:
    print(f"Error: {e}")
