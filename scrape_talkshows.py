#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:12:58 2024

@author: julianschibberges
"""

from bs4 import BeautifulSoup
import requests
import os
import json
import pandas as pd
from pytubefix import YouTube,Playlist
from datetime import datetime, timedelta
import locale
import hashlib
import base64
import re
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from collections import Counter


os.chdir("/Users/jschibberges/Documents/GitHub/wtmw")

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
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None
    except Exception as e:  # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        return None

all_annewill_data = load_json_file("./data/AnneWill_data.json")
all_carenmiosga_data = load_json_file("./data/CarenMiosga_data.json")
all_hartaberfair_data = load_json_file("./data/HartAberFair_data.json")
all_markuslanz_data = load_json_file("./data/MarkusLanz_data.json")
all_maischberger_data = load_json_file("./data/Maischberger_data.json")
all_illner_data = load_json_file("./data/Illner_data.json")

all_data = all_annewill_data+all_carenmiosga_data+all_hartaberfair_data+all_markuslanz_data+all_maischberger_data+all_illner_data
uids = [data["uid"] for data in all_data]

url_will = "https://daserste.ndr.de/annewill/archiv/"
url_miosga = "https://www.daserste.de/information/talk/caren-miosga/sendung/index.html"
url_hartaberfair = "https://www1.wdr.de/daserste/hartaberfair/sendungen/index.html"
url_maischberger = "https://www.daserste.de/information/talk/maischberger/sendung/index.html"
url_illner = "https://www.zdf.de/politik/maybrit-illner"


german_months = [
    "januar", "februar", "maerz", "april", "mai", "juni",
    "juli", "august", "september", "oktober", "november", "dezember"
]

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
    partei_strings = ["CDU", "CSU", "SPD", "freie wähler", "FDP", "BSW",
                      "B‘90/Grüne","Bündnis 90/ Die Grünen","AfD","Linke",
                      "parteilos", "Bündnis 90 / Die Grünen", "Bündnis 90/Die Grünen"]

    text_lower = text.lower().replace("("," ").replace(")"," ").replace(",", " ")
    # Collect items that are partial matches
    matches = [item for item in partei_strings if item.lower() in text_lower]
    if len(matches)==1:
        return matches[0]
    elif len(matches)==0:
        g_pattern =r"B[^.]{0,10}90[^.]*?[gG]rün\w{1,2}\b"
        matches = re.findall(g_pattern, text, flags=re.IGNORECASE)
        if "Die Grünen " in text:
            return "Bündnis 90 / Die Grünen"
        elif matches:
            return "Bündnis 90 / Die Grünen"
    else:
        return None

def clean_name(name:str):
    second_string=""
    new_name = None
    if "," in name and "(" in name:
        new_name = name.split(",")[0]
        second_string = " ".join(name.split(",")[1:])
    elif "," in name and "(" not in name:
        new_name = name.split(",")[0]
        second_string = " ".join(name.split(",")[1:])
    elif "," not in name and "(" in name:
        new_name = name.split("(")[0]
        second_string = " ".join(name.split("(")[1:])
    if new_name:
        if ":" in new_name:
            new_name = new_name.split(":")[1]
    elif ":" in name:
        name = name.split(":")[1]
    if new_name:
        new_name = new_name.strip()
        if second_string != "":
            party = extract_party(second_string)
            if party:
                return new_name, party
            else:
                return new_name, None
        else:
            return new_name, None     
    else:
        return name.strip(), None

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
                    date_object = datetime.strptime(date_string.replace(" ",""), '%d.%m.%Y')
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
    if not isinstance(dt_obj, datetime):
        return None  # Or raise a TypeError if you prefer

    previous_day = dt_obj - timedelta(days=1)
    return previous_day.strftime('%d.%m.%Y')




def extractShows_Will(base_url):
    def get_links(url):
        """Fetches all links containing 'https://daserste.ndr.de/annewill/archiv/' from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [
                link['href']
                for link in soup.find_all('a', href=True)
                if "/annewill/archiv/" in link['href']
            ]
            pre_url = "https://daserste.ndr.de"
            links = [pre_url+link for link in links if "index.html" not in link]
            return links
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return []

    # Step 1: Scrape links from the initial URL
    all_links = set(get_links(base_url))
    
    # Step 2: Find and process glossary pagination links
    pagination_links = [
        link for link in all_links if "erste318_glossaryPage" in link
    ]
    if pagination_links:
        pagination_links = sorted(pagination_links)[1:]  # Skip the first page
    
    # Step 3: Iterate through pagination links to scrape additional links
    for paginated_url in pagination_links:
        print(f"Scraping pagination URL: {paginated_url}")
        all_links.update(get_links(paginated_url))
    all_links = [link for link in all_links if "erste318_glossaryPage" not in link]
    return all_links


def extractShowsDetails_Will(links):
    all_data = []

    for link in links:
        try:
            uid = create_hash(link)
            if uid in uids:
                continue
            response = requests.get(link)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Initialize dictionary for the current link
            page_data = {}
            page_data["uid"] = uid
            # Scrape the title
            title_element = soup.find('h1', class_='headline small')
            page_data['title'] = title_element.get_text(strip=True) if title_element else None

            # Scrape the description
            description_element = soup.find('p', class_='text small')
            page_data['description'] = description_element.get_text(strip=True) if description_element else None

            # Scrape related broadcast details
            infokasten_element = soup.find('div', class_='infokasten relatedbroadcast small')
            if infokasten_element:
                h2_element = infokasten_element.find('h2')
                if h2_element:
                    broadcast_details = h2_element.get_text(strip=True).split('|')
                    if len(broadcast_details) == 4:
                        page_data['station'] = broadcast_details[0].strip()
                        page_data['show'] = broadcast_details[1].strip()
                        page_data['date'] = broadcast_details[2].strip()
                        page_data['time'] = broadcast_details[3].strip()
            page_data['link'] = link
            # Find the guest link
            guest_link_element = list(set([link['href'] for link in soup.find_all('a', href=True) if "Unsere-Gaeste" in link['href']]))
            pre_url = "https://daserste.ndr.de"
            guest_link_element = [pre_url+link for link in guest_link_element]
            if guest_link_element:
                guest_link = guest_link_element[0]
                page_data['guest_link'] = guest_link

                # Scrape guest details
                guest_response = requests.get(guest_link)
                guest_response.raise_for_status()
                guest_soup = BeautifulSoup(guest_response.text, 'html.parser')

                guests = []
                # Find all h3 elements with the class 'subtitle small'
                name_elements = guest_soup.find_all('h3', class_='subtitle small')
                
                for name_element in name_elements:
                    # Extract the name from the h3 element
                    name = name_element.get_text(strip=True)
                    name, party = clean_name(name)
                    # Find the next two p elements
                    role = None
                    description = None
                    
                    role_element = name_element.find_next('p')
                    if role_element:
                        role = role_element.get_text(strip=True)
                    if not party:
                        party = extract_party(role)
                    description_element = role_element.find_next('p') if role_element else None
                    if description_element:
                        description = description_element.get_text(strip=True)
                    
                    # Append the guest data as a dictionary
                    guests.append({
                        'name': name,
                        'party': party,
                        'role': role.strip(),
                        'description': description
                    })

                page_data['guests'] = guests

            all_data.append(page_data)

        except requests.RequestException as e:
            print(f"Error fetching URL {link}: {e}")
        except Exception as e:
            print(f"Error processing URL {link}: {e}")

    return all_data


def extractShows_Miosga(initial_url):
    base_url = "https://www.daserste.de/"
    urls = [initial_url]  # Initialize the list with the starting URL

    while True:
        current_url = urls[-1]  # Get the last URL in the list
        try:
            response = requests.get(current_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the next button link
            button_div = soup.find('div', class_='button left')
            if not button_div:
                break  # No more buttons, exit the loop

            next_link = button_div.find('a', href=True)
            if not next_link:
                break  # No valid link found, exit the loop

            # Prepend the base URL to the link
            next_url = base_url + next_link['href'].lstrip('/')
            if next_url in urls:  # Avoid infinite loops by checking for duplicates
                break

            urls.append(next_url)  # Add the new URL to the list

        except requests.RequestException as e:
            print(f"Error fetching URL {current_url}: {e}")
            break  # Stop if a request fails

    return urls




def extractShowsDetails_Miosga(links):
    all_data = []

    for link in links:
        try:
            uid = create_hash(link)
            if uid in uids:
                continue
            response = requests.get(link)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')

            # Initialize dictionary for the current page
            page_data = {}
            page_data["uid"] = uid

            # Scrape the title
            title_element = soup.find('h1', class_='headline small')
            page_data['title'] = title_element.get_text(strip=True) if title_element else None

            # Scrape the description
            description_element = soup.find('p', class_='text small')
            page_data['description'] = description_element.get_text(strip=True) if description_element else None

            # Scrape date, time, and station
            text_div = soup.find('div', class_='text')
            if text_div:
                text_element = text_div.contents[1].split("|")
                if len(text_element) >= 2:
                    page_data['date'] = text_element[0].split(",")[1].strip()
                    page_data['time'] = text_element[1].strip()
                    if len(text_element) >= 3:
                        page_data['station'] = text_element[2].strip()
            # Add a static "show" key
            page_data['show'] = "Caren Miosga"
            page_data["link"] = link

            # Scrape guests
            guests = []
            img_element = soup.find('img', alt="Nachgehakt")
            if img_element:
                name_elements = img_element.find_all_previous('h2', class_='subtitle small')
            else:
                name_elements = soup.find_all('h2', class_='subtitle small')

            for name_element in name_elements:
                name = name_element.get_text(strip=True)
                name, party = clean_name(name)
                # Extract the next p element as description
                description = None
                description_element = name_element.find_next('p')
                if description_element:
                    description = description_element.get_text(strip=True)

                # Extract the role from the span element
                role = None
                infotext_element = name_element.find_next("div").find('span', class_='infotext')
                if infotext_element:
                    infotext = infotext_element.get_text(strip=True)
                    if ',' in infotext and '|' in infotext:
                        role = infotext.split(',')[1].split('|')[0].strip()
                if not party:
                    if role is not None:
                        party = extract_party(role)
                # Append the guest data
                guests.append({
                    'name': name,
                    "party":party,
                    'role': role.strip(),
                    'description': description
                })
            
            page_data['guests'] = guests

            all_data.append(page_data)

        except requests.RequestException as e:
            print(f"Error fetching URL {link}: {e}")
        except Exception as e:
            print(f"Error processing URL {link}: {e}")

    return all_data


def extractShows_HartAberFair(initial_url):
    base_url = "https://www1.wdr.de"
    try:
        response = requests.get(initial_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [
            link['href']
            for link in soup.find_all('a', href=True)
            if "/hartaberfair/sendungen/" in link['href']
        ]
        links = [base_url+link for link in links if "index.html" not in link]
        return links
    except requests.RequestException as e:
        print(f"Error fetching URL {initial_url}: {e}")
        return []

def extractShowsDetails_HartAberFair(links):
    all_data = []
    
    for link in links:
        try:
            uid = create_hash(link)
            if uid in uids:
                continue
            response = requests.get(link)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
    
            # Initialize dictionary for the current page
            page_data = {}
            page_data["uid"] = uid

            # Scrape the title
            title_element = soup.find_all('h4', class_='headline')[1]
            page_data['title'] = title_element.get_text(strip=True) if title_element else None
    
            # Scrape the description
            description_element = title_element.find_next_sibling()
            page_data['description'] = description_element.get_text(strip=True) if description_element else None
            date_element = soup.find('h2', class_='conHeadline')
            page_data['date'] = date_element.get_text(strip=True).split(" ")[2]
            page_data['time'] = ""
            page_data['station'] = "Das Erste"
            page_data['show'] = "Hart Aber Fair"
            page_data["link"] = link

            # Scrape guests
            guests = []
            name_elements = soup.find('div', class_='modCon modConStage').find_all("h4", class_="headline")
            for name_element in name_elements:
                name = name_element.get_text(strip=True)
                name, party = clean_name(name)
                description = name_element.get("data-pre-headline").strip()
                if "," in description:
                    role = description.split(",")[0].strip()
                elif "und" in description:
                    role = description.split(" und ")[0].strip()
                else:
                    role = None
                if not party:
                    if role is not None:
                        party = extract_party(role)
                # Append the guest data
                guests.append({
                    'name': name,
                    'party':party,
                    'role': role.strip(),
                    'description': description
                })
            
            page_data['guests'] = guests
    
            all_data.append(page_data)
    
        except requests.RequestException as e:
            print(f"Error fetching URL {link}: {e}")
        except Exception as e:
            print(f"Error processing URL {link}: {e}")
    
    return all_data


def extractShows_Lanz(start_year=2):
    """
    Generate valid URLs for a TV show that airs on Tuesday, Wednesday, and Thursday,
    with German month names and replacement of special characters (e.g., ä -> ae).
    
    Args:
        start_year (int): Number of years to go back from the current year.
    
    Returns:
        list: A list of valid URLs.
    """
    base_url = "https://www.zdf.de/gesellschaft/markus-lanz/"
    # Set locale to German for month names
    locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
    
    # Character mapping for replacements
    char_replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss'
    }
    
    today = datetime.now()
    start_date = today - timedelta(days=start_year * 365)  # Approximate start date for 3 years ago
    valid_days = {1, 2, 3}  # Tuesday (1), Wednesday (2), Thursday (3)
    urls = []

    current_date = start_date
    while current_date <= today:
        if current_date.weekday() in valid_days:
            # Format the URL with German month names
            url_date = current_date.strftime("%-d-%B-%Y").lower()
            
            # Replace special German characters
            for original, replacement in char_replacements.items():
                url_date = url_date.replace(original, replacement)
            
            # Generate the full URL
            url = f"{base_url}markus-lanz-vom-{url_date}-100.html"
            urls.append(url)
        
        # Increment the day
        current_date += timedelta(days=1)
    
    return urls

def extractShowsDetails_Lanz(links):
    all_data = []
    
    for link in links:
        try:
            uid = create_hash(link)
            if uid in uids:
                continue
            response = requests.get(link)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
    
            # Initialize dictionary for the current page
            page_data = {}
            page_data["uid"] = uid

            # Scrape the title
            title_element = soup.find('h1', class_='big-headline')
            page_data['title'] = title_element.get_text(strip=True) if title_element else None
    
            # Scrape the description
            description = soup.find("p", class_="item-description")
            if description:
                page_data['description'] = description.get_text(strip=True)
            else:
                page_data['description'] = ""
            date_element = soup.find_all('dd', class_='teaser-info')[1]
            page_data['date'] = date_element.get_text(strip=True)
            page_data['time'] = ""
            page_data['station'] = "ZDF"
            page_data['show'] = "Markus Lanz"
            page_data["link"] = link

            # Scrape guests
            guests = []
            name_elements = soup.find('div', class_='b-post-content').find_all('b')
            for name_element in name_elements:
                namerole = name_element.get_text(strip=True).split(",")
                if len(namerole)==2:
                    name = namerole[0]
                    role = namerole[1]
                elif len(namerole)==1:
                    name = namerole[0]
                    role = None
                else:
                    name = namerole[0]
                    role = " ".join(namerole[1:])
                name, party = clean_name(name)
                if not party:
                    if role is not None:
                        party = extract_party(role)
                description = name_element.find_next_sibling(string=True)
                # Append the guest data
                guests.append({
                    'name': name,
                    'party':party,
                    'role': role.strip(),
                    'description': description
                })
            
            page_data['guests'] = guests
    
            all_data.append(page_data)
    
        except requests.RequestException as e:
            print(f"Error fetching URL {link}: {e}")
        except Exception as e:
            print(f"Error processing URL {link}: {e}")
    
    return all_data


def extractShows_Maischberger(initial_url):
    def get_data(url):
        """Fetches all links containing 'https://daserste.ndr.de/annewill/archiv/' from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            shows = soup.find_all("div", class_="box viewA")
            all_data = []
            links = []
            pre_url = "https://www.daserste.de"
            for show in shows:
                page_data = {}
        
                # Scrape the title
                title_element = show.find('h4', class_='headline')
                page_data['title'] = title_element.get_text(strip=True) if title_element else None
        
                # Scrape the description
                description_element = show.find("p", class_="teasertext")
                page_data['description'] = description_element.get_text(strip=True) if title_element else None
                title_split = page_data['title'].split(" am ")
                if len(title_split)==2:
                    page_data['show'] = title_split[0]
                    page_data['date'] = title_split[1]
                else:
                    page_data['show'] = "Maischberger"
                    page_data['date'] = show.find("h3", class_="ressort").string.split("vom ")[1].strip()
                page_data['time'] = ""
                page_data['station'] = "Das Erste"
                link_element = show.find("a", href=True)
                link = pre_url+link_element["href"]
                uid = create_hash(link)
                if uid in uids:
                    continue
                page_data["uid"] = create_hash(link)
                page_data["link"] = link
                links.append(link)
                response = requests.get(link)
                response.raise_for_status()
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, 'html.parser')
                extended_description = soup.find_all("p",class_="text small")
                if extended_description:
                    description = extended_description[0:len(extended_description)-2]
                    description = [des.get_text(strip=True) for des in description]
                    page_data['description'] = " ".join(description)
                specific_element = soup.find('h2', class_='subtitle small')
                
                if specific_element:
                    # Find all <p> elements that come after the specific element
                    paragraphs_after = specific_element.find_all_next('p', class_="text small")
                    guests = []
                    # Print the text of these paragraphs
                    pattern_com_par = r",\s*\("
                    pattern_par_com = r"\(\s*,"
                    for paragraph in paragraphs_after:
                        
                        guest_text = paragraph.get_text(strip=True)
                        if bool(re.search(pattern_com_par, guest_text)):
                            splits = guest_text.split(",")
                            name = splits[0]
                            if "(" in guest_text:
                                party = splits[1].split("(")[0]
                                role = splits[1].split("(")[1].replace(")","")
                            else:
                                match = extract_party(splits[1])
                                if match:
                                    party = match
                                    role = None
                                else:
                                    party = None
                                    role = splits[1]
                        elif bool(re.search(pattern_par_com, guest_text)): 
                            splits = guest_text.split("(")
                            name = splits[0]
                            role = splits[1].replace(")","")
                        else:
                            continue
                        description = ""
                        # Append the guest data
                        guests.append({
                            'name': name,
                            'party': party,
                            'role': role.strip(),
                            'description': description
                        })
                    page_data['guests'] = guests
                else:
                    page_data['guests'] = []
                
                all_data.append(page_data)
            return all_data, links
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return [],[]

    # Step 1: Scrape links from the initial URL
    all_data = []
    all_links = []
    response = requests.get(initial_url)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    button_element = soup.find("div", class_="button backToAll")
    last_page = button_element.find("div",class_="text").string.split("|")[1].strip()
    pagination_links = [initial_url]
    for i in range(1,int(last_page)):
        link = f"https://www.daserste.de/information/talk/maischberger/sendung/maischberger-sendungen-filter-100~_seite-{i}.html"
        pagination_links.append(link)

    # Step 3: Iterate through pagination links to scrape additional links
    for paginated_url in pagination_links:
        print(f"Scraping pagination URL: {paginated_url}")
        data, links = get_data(paginated_url)
        all_links.extend(links)
        all_data.extend(data)
    return all_links, all_data




def extractShows_Illner():
    base_url = "https://www.youtube.com/playlist?list=PLdPrKDvwrog5MvFTzlxs5L5QUazkvCeYa"
    p = Playlist(base_url)
    shows = []
    for url in p.video_urls:
        yt = YouTube(url)
        if yt.length > 1800:
            show = {"link":url,
                    "description":yt.description,
                    "title":yt.title,
                    "date":yt.publish_date}
            shows.append(show)
    return shows

def extractShowsDetails_Illner(shows):
    all_data = []
    for show in shows:
        uid = create_hash(show["link"])
        if uid in uids:
            continue
        page_data = {}
        page_data["uid"] = uid
        page_data["link"] = show["link"]
        page_data["description"] = show["description"]
        page_data["title"] = show["title"]
        page_data["date"] = subtract_date(show["date"])
        page_data["time"] = ""
        page_data["station"] = "ZDF"
        page_data["show"] = "Maybrit Illner"
        page_data["guests"] = []
        description = show["description"]
        if "Die Gäste der Sendung:" in description:
            guests = description.split("Die Gäste der Sendung:")[1].split("--")[0].split("__")[0]
            guests = guests.split("\n")
            for guest in guests:
                if guest.strip() != "":
                    if "," in guest:
                        name = guest.split(",")[0]
                        role = ",".join(guest.split(",")[1:])
                    elif "(" in guest:
                        name = guest.split("(")[0]
                        role = guest.split("(")[1].replace(")","")
                    name, party = clean_name(name)
                    if not party:
                        party = extract_party(role)
                    page_data["guests"].append({"name":name,"party":party,"role":role.strip(),"description":""})
        all_data.append(page_data)
    return all_data


# Scrape Anne Will
all_annewill_links = extractShows_Will(url_will)
print(f"Total links found: {len(all_annewill_links)}")
all_annewill_data.update(extractShowsDetails_Will(all_annewill_links))
#filename = f'{datetime.now().strftime("%Y%m%d")}_AnneWill_data.json'
filename = './data/AnneWill_data.json'
with open(filename, 'w') as f:
    json.dump(all_annewill_data, f)

# Scrape Caren Miosga
all_carenmiosga_links = extractShows_Miosga(url_miosga)
all_carenmiosga_data.update(extractShowsDetails_Miosga(all_carenmiosga_links))
#filename = f'{datetime.now().strftime("%Y%m%d")}_CarenMiosga_data.json'
filename = './data/CarenMiosga_data.json'
with open(filename, 'w') as f:
    json.dump(all_carenmiosga_data, f)

# Scrape Hart aber Fair
all_hartaberfair_links = extractShows_HartAberFair(url_hartaberfair)
all_hartaberfair_data.update(extractShowsDetails_HartAberFair(all_hartaberfair_links))
#filename = f'{datetime.now().strftime("%Y%m%d")}_HartAberFair_data.json'
filename = './data/HartAberFair_data.json'
with open(filename, 'w') as f:
    json.dump(all_hartaberfair_data, f)


# Scrape Markus Lanz
all_markuslanz_links = extractShows_Lanz(start_year=2)
all_markuslanz_data.update(extractShowsDetails_Lanz(all_markuslanz_links))
#filename = f'{datetime.now().strftime("%Y%m%d")}_MarkusLanz_data.json'
filename = './data/MarkusLanz_data.json'
with open(filename, 'w') as f:
    json.dump(all_markuslanz_data, f)

# Scrape Maischberger
all_maischberger_links, maischberger_data = extractShows_Maischberger(url_maischberger)
all_maischberger_data.update(maischberger_data)
#filename = f'{datetime.now().strftime("%Y%m%d")}_Maischberger_data.json'
filename = './data/Maischberger_data.json'
with open(filename, 'w') as f:
    json.dump(all_maischberger_data, f)


# Scrape Illner
all_illner_links = extractShows_Illner()
all_illner_data.update(extractShowsDetails_Illner(all_illner_links))
#filename = f'{datetime.now().strftime("%Y%m%d")}_Illner_data.json'
filename = './data/Illner_data.json'
with open(filename, 'w') as f:
    json.dump(all_illner_data, f)
    

# Aggregate
all_data = all_annewill_data+all_carenmiosga_data+all_hartaberfair_data+all_markuslanz_data+all_maischberger_data+all_illner_data
df = pd.DataFrame(all_data)
df.to_excel("all_data.xlsx", index=False)
guest_list = []
for dat in all_data:
    if "guests" in dat and dat["guests"] is not None:
        for guest in dat["guests"]:
            dic = {}
            dic.update(guest)
            dic["Talkshow"] = dat["show"]+" - "+str(dat["date"])
            guest_list.append(dic)
            if len(guest["name"])>37:
                print(dat["link"])
                print(guest["name"])


df_guests = pd.DataFrame(guest_list)
filename = f'{datetime.now().strftime("%Y%m%d")}_guests.xlsx'
df_guests.to_excel(filename)


grouped = df_guests.groupby('Talkshow')['name'].apply(list)

# Step 2: Generate edges
edges = []
for names in grouped:
    edges.extend(itertools.combinations(names, 2))  # Generate all pairs of people in the same Talkshow

# Count the occurrences of each pair
edge_counts = Counter(edges)

# Step 3: Create the graph with weighted edges
G = nx.Graph()
for edge, weight in edge_counts.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# Step 4: Visualize the network
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # Position nodes using the spring layout

# Draw nodes and edges
nx.draw(
    G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=2000, font_size=15
)

# Add edge labels (weights)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

plt.title("Co-occurrence Network of People in Talkshows (with Weights)")
plt.show()

# Step 5: Export the graph (optional)
nx.write_gexf(G, "cooccurrence_network_with_weights.gexf")

