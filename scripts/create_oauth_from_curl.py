"""
Create oauth.json from a cURL command.
Paste your cURL command when prompted.
"""
import json
import re

def extract_headers_from_curl(curl_command):
    """Extract headers from a cURL command."""
    headers = {}
    
    # Extract cookie
    cookie_match = re.search(r"-b '([^']+)'", curl_command)
    if cookie_match:
        headers['Cookie'] = cookie_match.group(1)
    
    # Extract other headers
    header_matches = re.findall(r"-H '([^:]+): ([^']+)'", curl_command)
    for key, value in header_matches:
        headers[key] = value
    
    # Extract user-agent
    ua_match = re.search(r"'user-agent: ([^']+)'", curl_command)
    if ua_match:
        headers['User-Agent'] = ua_match.group(1)
    
    return headers

def create_oauth_json(headers):
    """Create oauth.json from headers."""
    oauth_data = {
        "headers": {
            "User-Agent": headers.get('user-agent', headers.get('User-Agent', '')),
            "Accept": headers.get('accept', '*/*'),
            "Accept-Language": headers.get('accept-language', 'en-US,en;q=0.9'),
            "Content-Type": headers.get('content-type', 'application/json'),
            "X-Goog-AuthUser": headers.get('x-goog-authuser', '0'),
            "x-origin": headers.get('x-origin', 'https://music.youtube.com'),
            "Cookie": headers.get('Cookie', '')
        }
    }
    
    with open('oauth.json', 'w') as f:
        json.dump(oauth_data, f, indent=2)
    
    print("✅ Successfully created oauth.json!")
    return True

if __name__ == "__main__":
    print("Paste your cURL command below and press Enter:")
    print("-" * 60)
    
    curl_lines = []
    while True:
        try:
            line = input()
            if not line.strip():
                break
            curl_lines.append(line)
        except EOFError:
            break
    
    curl_command = ' '.join(curl_lines)
    
    if not curl_command.strip():
        print("❌ No cURL command provided")
        exit(1)
    
    headers = extract_headers_from_curl(curl_command)
    
    if not headers:
        print("❌ Could not extract headers from cURL command")
        exit(1)
    
    create_oauth_json(headers)
    print("\nYou can now use the YouTube Music tool in your voice assistant!")
