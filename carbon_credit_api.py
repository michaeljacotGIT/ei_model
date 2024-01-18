from bs4 import BeautifulSoup
import requests
import re


def get_website_body_as_string(url, debug=False):
    # Read the URL from a json file for security reasons as per user guidelines
    # Make sure to create a json file with the API keys or passwords if needed
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            body_content = soup.body.get_text(separator=' ', strip=True)
            if debug:
                print(f"URL fetched successfully: {url}")
                print(f"Status Code: {response.status_code}")
                print("Body content extracted")
            return body_content
        else:
            if debug:
                print(f"Failed to fetch URL: {url}")
                print(f"Status Code: {response.status_code}")
            return ""
    except Exception as e:
        if debug:
            print(f"An error occurred: {e}")
        return ""

def find_eu_ets_price(text, debug=False):
    # Regex pattern to find "EU Carbon Permits" followed by a price like pattern
    pattern = r"EU Carbon Permits\s+(\d+\.\d+)"

    try:
        # Search for the pattern in the provided text
        match = re.search(pattern, text)

        # If a match is found, return the price
        if match:
            price = match.group(1)
            if debug:
                print(f"EU-ETS Carbon Credit price found: {price}")
            return price
        else:
            if debug:
                print("EU-ETS Carbon Credit price not found.")
            return None
    except Exception as e:
        if debug:
            print(f"An error occurred: {e}")
        return None


def getETUPrice():
    try:
        url = "https://tradingeconomics.com/commodity/carbon"  # Replace with your actual URL
        body_string = get_website_body_as_string(url, debug=True)
        price = find_eu_ets_price(body_string, debug=True)

        if price is not None:
            # Convert price to USD from EUR
            price_usd = round(float(price) * 1.18, 2)
        else:
            # If price is not found, use the default value
            price_usd = 93.37

    except Exception as e:
        print(f"An error occurred: {e}")
        # Use the default value in case of any error
        price_usd = 93.37

    return price_usd



print(getETUPrice())