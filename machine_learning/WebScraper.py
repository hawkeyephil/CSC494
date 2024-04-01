#Import Statements
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path 

#Relative filepath to .csv file with URLs to FED Statements/Dates 
path = Path(__file__).parent.parent / "CleanedPolicyStatementURLs.csv" 
#Pandas Dataframe that contains all of the URLs and dates 
dataframe = pd.read_csv(path, sep = ',') 

#Web Scraper Loop 
for index, row in dataframe.iterrows(): 
    #URL to request webpage 
    url = row[0] 
    #Debugging 
    print('Scraping: ' + url)

    #Date of FED statement 
    numericDate = row[1]
    date = str(numericDate)
    year = date[0:4] 
    month = date[4:6]
    day = date[6:8]
    date = year + '-' + month + '-' + day 
    #Debugging 
    print('FED Statement Release:' + date) 

    #Request webpage and create BeautifulSoup object to parse html 
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser") 

    #Pre 2006 webpages 
    if(numericDate < 20060000): 
        #Parses and stores all text with <p> tag 
        paragraphs = soup.find_all('p') 
        data = ''
        for p in paragraphs: 
            data = data + '\n' + p.text 
    #Post 2005 webpages 
    else: 
        #Collects publish date 
        specific_p_tags = soup.find_all('p', class_=['article__time'])

        #Locates the specific <div> with the class="col-xs-12 col-sm-8 col-md-8"
        target_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8') 

        #Collects all <p> tags within the target <div>
        p_tags = target_div.find_all('p') 

        #Extracts publish date 
        time = ''
        for p in specific_p_tags: 
            time = p.text

        #Appends publish date and then extracts/appends paragraphs from the statements 
        data = time 
        for p in p_tags: 
            data = data + '\n' + p.text 

    #Unique filename for each statement saved as a .txt file 
    filename = 'FEDStatements/' + date + '.txt' 
    #Relative path to FEDStatements folder to store each statment within the project 
    path = Path(__file__).parent / filename 
    #Writes the contents of the statement to a .txt file and throws an error message if an exception occurs 
    try:
        with open(path, 'w', encoding='UTF-8') as file: 
            file.write(data)
    except FileNotFoundError:
        print("Error") 
