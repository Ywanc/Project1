from bs4 import BeautifulSoup
import requests
import pandas as pd
from urllib.parse import urljoin


keywords = ['facts', 'background']

# check whether a judgment(bsoup) has a "Facts" main header
def has_facts_header(soup):
    headings = soup.find_all(class_=lambda x:x and 'Judg-Heading-1' in x)
    for heading in headings:
        if any(keyword in heading.text.lower() for keyword in keywords):
            return True
    return False

# adds judgement and hacts_header(boolean) to csv file
def scrape_to_csv(url, csv_filepath):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    has_facts = has_facts_header(soup)
    sample = pd.DataFrame([[url.split('/')[-1], has_facts]], columns=['case_number', 'has_facts_header'])
    sample.to_csv(csv_filepath, mode='a', header=False, index=False)

# fetch all cases from a Elit page
def fetch_case_links(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tags = soup.find_all('a', class_='gd-heardertext')
    case_links = [urljoin(base_url, tag.get("href")) for tag in tags]
    return case_links
    

url1 = "https://www.elitigation.sg/gd/Home/Index?Filter=SUPCT&YearOfDecision=All&SortBy=DateOfDecision&CurrentPage=" 
url2 = "&SortAscending=False&PageSize=0&Verbose=False&SearchQueryTime=0&SearchTotalHits=0&SearchMode=True&SpanMultiplePages=False"
for i in range(1, 981):
    page_links = fetch_case_links(url1 + str(i) + url2)
    print(f"Scraping {i}th page ... ")
    for link in page_links:
        scrape_to_csv(link, "eda_dataset.csv")