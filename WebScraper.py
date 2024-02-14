import requests
from bs4 import BeautifulSoup

import numpy as np

# Assuming your CSV file is named 'grades.csv'
arr = np.loadtxt('C:\\Users\\Philip Caldarella\\Desktop\\CSC494\\PolicyStatementUrls.csv', delimiter=',', dtype=str)

# Now 'arr' contains the data from the CSV file
print("The array is:")
print(arr)


# Fetch the FOMC statement page 
for page in arr:
    url = page 
    year = page[64:68]
    month = page[68:70]
    day = page[70:72] 
    date = year + '-' + month + '-' + day
    print(date)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser") 
    specific_p_tags = soup.find_all('p', class_=['article__time'])#, 'releaseTime'])

        # Find the specific <div> with the class "col-xs-12 col-sm-8 col-md-8"
    target_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8') 

        # Extract all <p> tags within the target <div>
    p_tags = target_div.find_all('p')

        # Extract all <p> tags
        #paragraphs = soup.find_all("p")

        # Print the text content of each <p> tag
        #for p in paragraphs:
        #    print(p.text)

    time = ''
    for p in specific_p_tags: 
        #print(p.text) 
        time = p.text

    data = time 
    for p in p_tags: 
        #print(p.text) 
        data = data + '\n' + p.text

    filename = 'C:\\Users\\Philip Caldarella\\Desktop\\CSC494\\Policy Statements\\' + date + '.txt'
    #print(data)
    try:
        with open(filename, 'w', encoding='UTF-8') as file: 
            #print('HI')
            file.write(data)
    except FileNotFoundError:
        print("Error") 
