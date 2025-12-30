"""
Download Delta Airlines policy pages for offline processing.
"""
import requests
from pathlib import Path
from src.config import DATA_DIR

DATA_DIR.mkdir(exist_ok=True)

# Request headers to mimic browser behavior
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Delta Airlines policy URLs
POLICY_URLS = {
    "delta_domestic_contract.html": "https://www.delta.com/us/en/legal/contract-of-carriage-dgr",
    "delta_international_contract.html": "https://www.delta.com/us/en/legal/contract-of-carriage-igr",
    "delta_baggage_faqs.html": "https://www.delta.com/us/en/baggage/additional-baggage-information/baggage-faqs"
}


def download_policies():
    """Download policy HTML files to data directory."""
    success_count = 0
    
    for filename, url in POLICY_URLS.items():
        print(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            
            file_path = DATA_DIR / filename
            file_path.write_text(response.text, encoding="utf-8")
            
            size_kb = len(response.text) // 1000
            print(f"  Saved: {size_kb} KB")
            success_count += 1
            
        except requests.RequestException as e:
            print(f"  Error: {e}")
        except IOError as e:
            print(f"  File error: {e}")
    
    print(f"\nDownload complete: {success_count}/{len(POLICY_URLS)} files")
    
    if success_count < len(POLICY_URLS):
        print("Warning: Some files failed. System will work with partial data.")
    
    return success_count


if __name__ == "__main__":
    download_policies()
