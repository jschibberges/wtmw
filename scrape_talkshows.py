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
from datetime import datetime, timedelta
import locale


os.chdir("/Users/julianschibberges/Library/CloudStorage/OneDrive-BernsteinGroup/talkshows")

url_will = "https://daserste.ndr.de/annewill/archiv/"
url_miosga = "https://www.daserste.de/information/talk/caren-miosga/sendung/index.html"
url_hartaberfair = "https://www1.wdr.de/daserste/hartaberfair/sendungen/index.html"
url_maischberger = "https://www.daserste.de/information/talk/maischberger/sendung/index.html"

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
            response = requests.get(link)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Initialize dictionary for the current link
            page_data = {}

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
            
                    # Find the next two p elements
                    role = None
                    description = None
                    
                    role_element = name_element.find_next('p')
                    if role_element:
                        role = role_element.get_text(strip=True)
                    
                    description_element = role_element.find_next('p') if role_element else None
                    if description_element:
                        description = description_element.get_text(strip=True)
                    
                    # Append the guest data as a dictionary
                    guests.append({
                        'name': name,
                        'role': role,
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
            response = requests.get(link)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')

            # Initialize dictionary for the current page
            page_data = {}

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
                if len(text_element) >= 3:
                    page_data['date'] = text_element[0].split(",")[1].strip()
                    page_data['time'] = text_element[1].strip()
                    page_data['station'] = text_element[2].strip()
                    # Add a static "show" key
                    page_data['show'] = "Caren Miosga"

            # Scrape guests
            guests = []
            name_elements = soup.find_all('h2', class_='subtitle small')
            for name_element in name_elements:
                name = name_element.get_text(strip=True)
                
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

                # Append the guest data
                guests.append({
                    'name': name,
                    'role': role,
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
            response = requests.get(link)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
    
            # Initialize dictionary for the current page
            page_data = {}
    
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
            
            # Scrape guests
            guests = []
            name_elements = soup.find('div', class_='modCon modConStage').find_all("h4", class_="headline")
            for name_element in name_elements:
                name = name_element.get_text(strip=True)
                description = name_element.get("data-pre-headline").strip()
                if "," in description:
                    role = description.split(",")[0].strip()
                elif "und" in description:
                    role = description.split("und")[0].strip()
                else:
                    role = ""
                # Append the guest data
                guests.append({
                    'name': name,
                    'role': role,
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
            response = requests.get(link)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
    
            # Initialize dictionary for the current page
            page_data = {}
    
            # Scrape the title
            title_element = soup.find('h1', class_='big-headline')
            page_data['title'] = title_element.get_text(strip=True) if title_element else None
    
            # Scrape the description
            page_data['description'] = ""
            date_element = soup.find_all('dd', class_='teaser-info')[1]
            page_data['date'] = date_element.get_text(strip=True)
            page_data['time'] = ""
            page_data['station'] = "ZDF"
            page_data['show'] = "Markus Lanz"
            
            # Scrape guests
            guests = []
            name_elements = soup.find('div', class_='b-post-content').find_all('b')
            for name_element in name_elements:
                namerole = name_element.get_text(strip=True).split(",")
                name = namerole[0]
                role = namerole[1]
                description = name_element.find_next_sibling(string=True)
                # Append the guest data
                guests.append({
                    'name': name,
                    'role': role,
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
                links.append(link_element["href"])
                # Scrape guests
                guests = []
                guest_text = page_data["description"].replace("|","").replace("Zu Gast:","").replace(") und ",");").replace("),",");").replace(").",");").strip().split(");")
                del guest_text[-1]
                for guest in guest_text:
                    namerole = guest.split("(")
                    name = namerole[0].strip()
                    role = namerole[1].strip()
                    description = ""
                    # Append the guest data
                    guests.append({
                        'name': name,
                        'role': role,
                        'description': description
                    })
                
                page_data['guests'] = guests
                all_data.append(page_data)

            links = [pre_url+link for link in links if "index.html" not in link]
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



# Scrape Anne Will
all_annewill_links = extractShows_Will(url_will)
print(f"Total links found: {len(all_annewill_links)}")
all_annewill_data = extractShowsDetails_Will(all_annewill_links)
filename = f'{datetime.now().strftime("%Y%m%d")}_AnneWill_data.json'
with open(filename, 'w') as f:
    json.dump(all_annewill_data, f)

# Scrape Caren Miosga
all_carenmiosga_links = extractShows_Miosga(url_miosga)
all_carenmiosga_data = extractShowsDetails_Miosga(all_carenmiosga_links)
filename = f'{datetime.now().strftime("%Y%m%d")}_CarenMiosga_data.json'
with open(filename, 'w') as f:
    json.dump(all_carenmiosga_data, f)

# Scrape Hart aber Fair
all_hartaberfair_links = extractShows_HartAberFair(url_hartaberfair)
all_hartaberfair_data = extractShowsDetails_HartAberFair(all_hartaberfair_links)
filename = f'{datetime.now().strftime("%Y%m%d")}_HartAberFair_data.json'
with open(filename, 'w') as f:
    json.dump(all_hartaberfair_data, f)


# Scrape Markus Lanz
all_markuslanz_links = extractShows_Lanz(start_year=2)
all_markuslanz_data = extractShowsDetails_Lanz(all_markuslanz_links)
filename = f'{datetime.now().strftime("%Y%m%d")}_MarkusLanz_data.json'
with open(filename, 'w') as f:
    json.dump(all_markuslanz_data, f)



# Scrape Maischberger
all_maischberger_links, all_maischberger_data = extractShows_Maischberger(url_maischberger)
filename = f'{datetime.now().strftime("%Y%m%d")}_Maischberger_data.json'
with open(filename, 'w') as f:
    json.dump(all_maischberger_data, f)


# Scrape Illner




# Aggregate
all_data = all_annewill_data+all_carenmiosga_data+all_hartaberfair_data+all_markuslanz_data+all_maischberger_data
df = pd.DataFrame(all_data)
guest_list = []
for dat in all_data:
    if "guests" in dat:
        guest_list.extend(dat["guests"])

df_guests = pd.DataFrame(guest_list)
df_guests.to_excel("guests.xlsx")
