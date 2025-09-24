#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
from datetime import datetime, timedelta, date
import locale
import hashlib
import base64
import re
from pathlib import Path


# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

uids: list = []

def load_json_file(filepath):
    """Loads data from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns: 
        The loaded JSON data as a Python dictionary or list, or None if 
        an error occurs during loading.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # Explicitly handle encoding
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return []
    except Exception as e:  # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        return []

def save_json_file(filepath, data):
    """Saves data to a JSON file.

    Args:
        filepath: The path to the JSON file.
        data: The Python dictionary or list to save.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved data to {filepath}")
    except Exception as e:
        print(f"An error occurred while saving to {filepath}: {e}")

all_annewill_data = load_json_file("./data/AnneWill_data.json")
all_carenmiosga_data = load_json_file("./data/CarenMiosga_data.json")
all_hartaberfair_data = load_json_file("./data/HartAberFair_data.json")
all_markuslanz_data = load_json_file("./data/MarkusLanz_data.json")
all_maischberger_data = load_json_file("./data/Maischberger_data.json")
all_illner_data = load_json_file("./data/Illner_data.json")

all_data = all_annewill_data+all_carenmiosga_data+all_hartaberfair_data+all_markuslanz_data+all_maischberger_data+all_illner_data
all_data = [dat for dat in all_data if dat is not None]

uids = {data["uid"] for data in all_data if "uid" in data}


alternative_url_will = "https://www.fernsehserien.de/anne-will/episodenguide"
alternative_url_miosga = "https://www.fernsehserien.de/caren-miosga/episodenguide"
alternative_url_hartaberfair = "https://www.fernsehserien.de/hart-aber-fair/episodenguide"
alternative_url_maischberger = "https://www.fernsehserien.de/maischberger-ard/episodenguide"
alternative_url_illner = "https://www.fernsehserien.de/maybrit-illner/episodenguide"
alternative_url_markuslanz = "https://www.fernsehserien.de/markus-lanz/episodenguide"

PARTY_STRINGS = ["CDU", "CSU", "SPD", "freie wähler", "FDP", "BSW", "AfD", "Linke", "parteilos"]
PARTY_NORMALIZATION_MAP = {
    # Variations for "Bündnis 90/Die Grünen"
    "B‘90/Grüne": "Bündnis 90/Die Grünen",
    "Bündnis 90/ Die Grünen": "Bündnis 90/Die Grünen",
    "Bündnis 90/Die Grünen": "Bündnis 90/Die Grünen",
    "B'90/Grüne": "Bündnis 90/Die Grünen",
    "B´90/Die Grünen": "Bündnis 90/Die Grünen",
    "B´90/Grüne": "Bündnis 90/Die Grünen",
    "Bündnis 90 / Die Grünen": "Bündnis 90/Die Grünen",
    # Regex patterns can also be pre-compiled here if needed
}
GREEN_PARTY_PATTERN = re.compile(r"B[^.]{0,10}90[^.]*?[gG]rün\w{1,2}\b", flags=re.IGNORECASE)

# --- Utility Functions ---
def create_hash(url, length=8):
    """
    Create a unique and short hash for a given URL.
    
    Args:
        url (str): The URL to hash.
        length (int): The desired length of the short hash. Default is 8.
    
    Returns:
        str: A short hash of the URL.
    """
    # Generate a SHA256 hash of the URL
    sha256_hash = hashlib.sha256(url.encode()).digest()
    # Encode the hash in Base64 for shorter representation
    base64_hash = base64.urlsafe_b64encode(sha256_hash).decode()
    # Truncate to the desired length
    return base64_hash[:length]

def extract_party(text):
    """
    Extract items from the list that are partial matches in the string (case insensitive).
    
    Args:
        text (str): The string to check against.
    
    Returns:
        string: The matching string from the text.
    """
    if not text:
        return None

    # Normalize common variations
    for variation, canonical in PARTY_NORMALIZATION_MAP.items():
        if variation in text:
            return canonical

    # Check for standard party names
    text_lower = text.lower()
    for party in PARTY_STRINGS:
        if party.lower() in text_lower:
            return party

    # Check for regex patterns for "Grüne"
    if GREEN_PARTY_PATTERN.search(text) or "Die Grünen" in text:
        return "Bündnis 90/Die Grünen"

    return None

def clean_name(name:str):
    """
    Cleans a guest's name string, removing titles and extracting party affiliation.
    """
    if not name:
        return "", None

    # Remove titles
    name = name.replace("Prof.","").replace("Dr.","").strip()

    # Split name from other info (party, role)
    # Use regex to find the first occurrence of a separator
    match = re.search(r'[,(:]', name)
    if match:
        separator_index = match.start()
        main_name = name[:separator_index].strip()
        other_info = name[separator_index:].strip()
    else:
        main_name = name
        other_info = ""

    # Clean up name from leading colons if any
    if ":" in main_name:
        main_name = main_name.split(":")[-1].strip()

    party = extract_party(other_info)

    return main_name, party

def standardize_date(date_string):
    """
    Standardizes date strings to DD.MM.YYYY format.

    Args:
        date_string: The date string to standardize.

    Returns:
        A standardized date string in DD.MM.YYYY format, or None if the 
        input is invalid or cannot be parsed.
    """
    locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')  # Set locale for German month names

    try:
        # Attempt parsing with different formats
        try:
            # Format 1: DD.MM.YYYY
            date_object = datetime.strptime(date_string, '%d.%m.%Y')
        except ValueError:
            try:
                # Format 2: DD. Month YYYY
                date_object = datetime.strptime(date_string, '%d. %B %Y')
            except ValueError:
                try:
                   # Format 3: DD.MM.YYYY (with potential space after the day)
                    date_object = datetime.strptime(date_string.replace(" ", ""), '%d.%m.%Y')
                except ValueError:
                    return None  # Invalid format

        return date_object.strftime('%d.%m.%Y')

    except Exception as e:  # Catches other potential errors (e.g., TypeError)
        print(f"Error parsing date: {e}")
        return None

def subtract_date(dt_obj):
    """
    Formats a datetime.datetime object to DD.MM.YYYY after subtracting one day.

    Args:
        dt_obj: A datetime.datetime object

    Returns:
         A string representing the date in DD.MM.YYYY format, one day
         before the provided datetime object.  Returns None if input is not a 
         datetime object.
    """
    if not isinstance(dt_obj, (datetime, date)):
        return None  # Or raise a TypeError if you prefer

    previous_day = dt_obj - timedelta(days=1)
    return previous_day.strftime('%d.%m.%Y')


def get_episode_urls_from_guide(guide_url):
    """
    Scrapes an episode guide page to find all individual episode URLs.

    Args:
        guide_url (str): The URL of the episode guide page.

    Returns:
        list: A list of unique URLs for individual episodes.
    """
    base_url = "https://www.fernsehserien.de"
    episode_urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(guide_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # The episode rows are <a> tags with itemprop="episode".
        # We find all of them directly, as they can be in multiple list containers.
        episode_rows = soup.find_all('a', itemprop='episode')
        
        for row in episode_rows:
            relative_link = row.get('href')
            if relative_link:
                episode_urls.append(base_url + relative_link)
                
    except requests.RequestException as e:
        print(f"Error fetching guide URL {guide_url}: {e}")
    
    return list(set(episode_urls)) # Return unique URLs

def get_episode_details(episode_urls: list) -> list:
    """
    Scrapes a single episode page for its details.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    cnt_max = len(episode_urls)
    cnt = 0
    episode_data = []
    for episode_url in episode_urls:
        cnt += 1
        try:
            uid = create_hash(episode_url)
            if uids:
                if uid in uids:
                    continue # Already scraped
            else:
                uids.append(uid)

            print(f"  Scraping episode {cnt}/{cnt_max}: {episode_url}")
            episode_page_response = requests.get(episode_url, headers=headers)
            episode_page_response.raise_for_status()
            episode_soup = BeautifulSoup(episode_page_response.content, 'html.parser')

            episode_info = {"uid": uid, "link": episode_url, "guests": []}

            # --- Date Extraction and Check ---
            episode_date_obj = None
            broadcast_container = episode_soup.find('ea-angaben')
            if not broadcast_container:
                print(f"    Skipping episode, no broadcast info container ('ea-angaben') found on page.")
                continue
            
            date_tag = broadcast_container.find('ea-angabe-datum')
            if date_tag:
                date_text = date_tag.get_text(strip=True)
                # The date can be "So. 01.12.2024" or just "01.12.2024"
                date_match = re.search(r'(\d{1,2}\.\d{1,2}\.\d{4})', date_text)
                if date_match:
                    date_str = date_match.group(1)
                    date_str_standardized = standardize_date(date_str)
                    if date_str_standardized:
                        try:
                            episode_date_obj = datetime.strptime(date_str_standardized, '%d.%m.%Y').date()
                        except (ValueError, TypeError):
                            print(f"    Could not parse date from '{date_str}'")
            
            # If date is in the future or not found, skip this episode
            if not episode_date_obj or episode_date_obj > date.today():
                if episode_date_obj:
                    print(f"    Skipping future episode from {episode_date_obj.strftime('%d.%m.%Y')}")
                else:
                    print("    Skipping episode, no valid date found.")
                continue
            
            episode_info['date'] = episode_date_obj.strftime('%d.%m.%Y')
            
            # Show name from breadcrumbs
            # Show name from h1.serien-titel, which is more robust than breadcrumbs
            show_name_h1 = episode_soup.find('h1', class_='serien-titel')
            if show_name_h1 and show_name_h1.a:
                episode_info['show'] = show_name_h1.a.get_text(strip=True)
            else:
                episode_info['show'] = None

            # Station name
            station_tag = broadcast_container.find('ea-angabe-sender')
            if station_tag:
                episode_info['station'] = station_tag.get_text(strip=True)
            else:
                episode_info['station'] = None

            # Extract title
            # The site uses different tags for the title (h1 for new, h3 for old pages), so we check for both.
            episode_title_tag = episode_soup.find('h1', class_='episode-title')
            if not episode_title_tag:
                episode_title_tag = episode_soup.find('h3', class_='episode-output-titel')
                
            if episode_title_tag and episode_title_tag.find('span', itemprop='name'):
                episode_info['title'] = episode_title_tag.find('span', itemprop='name').get_text(strip=True)
            else:
                episode_info['title'] = None


            # Extract description from the main content area
            inhalt_div = episode_soup.find('div', class_='episode-output-inhalt-inner')
            if inhalt_div:
                # Create a copy to manipulate without affecting other scraping logic
                inhalt_clone = BeautifulSoup(str(inhalt_div), 'html.parser')

                # Remove ad-related tags, source tags, and other noise
                for tag in inhalt_clone.find_all(['span', 'werbung', 'ins', 'script']):
                    tag.decompose()

                # Use a unique separator to split lines later
                full_text = inhalt_clone.get_text(separator='|||', strip=True)
                
                # Split the text by the guest introduction
                parts = re.split(r'Die Gäste:|||', full_text, maxsplit=1, flags=re.IGNORECASE)
                
                description_parts = []
                
                # Part before "Die Gäste:" is likely description
                if parts[0]:
                    description_parts.append(parts[0])
                    
                # If there is a part after "Die Gäste:", find where the description continues
                if len(parts) > 1:
                    lines_after_guests = [line.strip() for line in parts[1].split('|||') if line.strip()]
                    for i, line in enumerate(lines_after_guests):
                        # Heuristic: A description is a longer sentence. A guest entry is short.
                        if line and (len(line.split()) > 10 or line.endswith('.') or line.endswith('?')):
                            description_parts.extend(lines_after_guests[i:])
                            break
                
                episode_info['description'] = ' '.join(description_parts).replace('|||', ' ').strip()
            else:
                episode_info['description'] = ""
            if "Die Gäste: " in episode_info.get('description'):
                episode_info['description'] = episode_info['description'].split("Die Gäste: ")[0].strip()
            # Extract guests from the "Cast & Crew" section
            cast_crew_list = episode_soup.find('ul', class_='cast-crew')
            if cast_crew_list:
                guest_items = cast_crew_list.find_all('li', itemscope=True, itemtype="http://schema.org/Person")
                for item in guest_items:
                    dd_tag = item.find('dd')
                    if dd_tag and dd_tag.p:
                        # Use stripped_strings to handle <br> tags and get clean text parts
                        p_strings = list(dd_tag.p.stripped_strings)
                        if p_strings and p_strings[0].lower() == 'gast':
                            name_tag = item.find('dt', itemprop='name')
                            if name_tag:
                                name = name_tag.get_text(strip=True)
                                # The rest of the strings form the description
                                description = " ".join(p_strings[1:]) if len(p_strings) > 1 else ""
                                
                                cleaned_name, party = clean_name(name)
                                if not party:
                                    party = extract_party(description)
                                episode_info['guests'].append({
                                    "name": cleaned_name, "party": party, "role": description, "description": description
                                })
            
            episode_data.append(episode_info)

        except requests.RequestException as e:
            print(f"Error fetching URL {episode_url}: {e}")
        except Exception as e:
            print(f"An error occurred while scraping {episode_url}: {e}")
        
        continue
    return episode_data

def scrape_fernsehserien_episodeguide(url):
    """
    Scrapes episode data from a fernsehserien.de episodenguide URL.

    This function first scrapes the main episode guide page to get links to
    individual episode pages. Then, it visits each episode page to extract
    detailed information, including guests.

    Args:
        url (str): The URL of the episodenguide page.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              metadata for an episode.
    """
    print(f"Scraping episode guide: {url}")
    episode_urls = get_episode_urls_from_guide(url)
    print(f"Found {len(episode_urls)} total episode links.")
    
    all_show_data = get_episode_details(episode_urls)
            
    return all_show_data

def main():
    """Main function to run the scraping process."""
    DATA_DIR.mkdir(exist_ok=True)

    # Using fernsehserien.de scraper
    print("\n-- - Scraping Caren Miosga from fernsehserien.de --")
    miosga_fernsehserien_data = scrape_fernsehserien_episodeguide(alternative_url_miosga)
    if miosga_fernsehserien_data:
        all_carenmiosga_data.extend(miosga_fernsehserien_data)
        save_json_file(DATA_DIR / 'CarenMiosga_data.json', all_carenmiosga_data)
    
    print("\n-- - Scraping Anne Will from fernsehserien.de --")
    will_fernsehserien_data = scrape_fernsehserien_episodeguide(alternative_url_will)
    if will_fernsehserien_data:
        all_annewill_data.extend(will_fernsehserien_data)
        save_json_file(DATA_DIR / 'AnneWill_data.json', all_annewill_data)
    
    print("\n-- - Scraping Hart aber Fair from fernsehserien.de --")
    hartaberfair_fernsehserien_data = scrape_fernsehserien_episodeguide(alternative_url_hartaberfair)
    if hartaberfair_fernsehserien_data:
        all_hartaberfair_data.extend(hartaberfair_fernsehserien_data)
        save_json_file(DATA_DIR / 'HartAberFair_data.json', all_hartaberfair_data)
    
    print("\n-- - Scraping Maischberger from fernsehserien.de --")
    maischberger_fernsehserien_data = scrape_fernsehserien_episodeguide(alternative_url_maischberger)
    if maischberger_fernsehserien_data:
        all_maischberger_data.extend(maischberger_fernsehserien_data)
        save_json_file(DATA_DIR / 'Maischberger_data.json', all_maischberger_data)
    
    print("\n-- - Scraping Markus Lanz from fernsehserien.de --")
    markuslanz_fernsehserien_data = scrape_fernsehserien_episodeguide(alternative_url_markuslanz)
    if markuslanz_fernsehserien_data:
        all_markuslanz_data.extend(markuslanz_fernsehserien_data)
        save_json_file(DATA_DIR / 'MarkusLanz_data.json', all_markuslanz_data)
    
    print("\n-- - Scraping Maybrit Illner from fernsehserien.de --")
    illner_fernsehserien_data = scrape_fernsehserien_episodeguide(alternative_url_illner)
    if illner_fernsehserien_data:
        all_illner_data.extend(illner_fernsehserien_data)
        save_json_file(DATA_DIR / 'Illner_data.json', all_illner_data)

    # Aggregate all data and save to Excel
    print("\n-- - Aggregating all data --")
    all_data_final = (all_annewill_data + all_carenmiosga_data + all_hartaberfair_data +
                      all_markuslanz_data + all_maischberger_data + all_illner_data)
    
    all_data_final2 = (miosga_fernsehserien_data + will_fernsehserien_data +
                        hartaberfair_fernsehserien_data + maischberger_fernsehserien_data +
                        markuslanz_fernsehserien_data + illner_fernsehserien_data)
    
    # Deduplicate based on UID
    unique_data = {item['uid']: item for item in all_data_final}.values()
    df = pd.DataFrame(list(unique_data))
    df.to_excel(DATA_DIR / "all_data.xlsx", index=False)
    print("Aggregation complete. Excel file saved.")

    df = pd.DataFrame(all_data_final2)
    df.to_excel(DATA_DIR / "all_data2.xlsx", index=False)
    print("Aggregation complete. Excel file saved.")

if __name__ == "__main__":
    main()