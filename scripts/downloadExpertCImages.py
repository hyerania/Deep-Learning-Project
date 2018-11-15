from bs4 import BeautifulSoup
import requests
import re
import os

html_page = requests.get("https://data.csail.mit.edu/graphics/fivek/")

text = html_page.text

# print(text)
soup = BeautifulSoup(text, 'html.parser')
images = []

for link in soup.findAll('a'):
    # print(link.get('href'))
    images.append(link.get('href'))


print(images)
print(len(images))
for image in images[14000:]:
    if image is not None:
        filename = image.split('/')
        if 'tif' in filename[-1]:
            if 'tiff16_c' in filename[-2]:
                file_name = filename[-2] + "_" + filename[-1]
                print(filename)
                print(file_name)
                url = "https://data.csail.mit.edu/graphics/fivek/" + image
                # print(url)
                r = requests.get(url, allow_redirects=True)
                open(file_name, 'wb').write(r.content)
