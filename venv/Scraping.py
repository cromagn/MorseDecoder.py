import requests
import os
from bs4 import BeautifulSoup
import shutil
myfile = open("c://tmp/immagini_full.txt", 'w')
for x in range (1500):
    page = requests.get("https://network.satnogs.org/stations/"+str(x)+"/")
    soup = BeautifulSoup(page.content, 'html.parser')
    info=soup.find_all(class_='station-view-image')
    print(len(info))
    if len(info)>0:
        desc = str(info[0]['alt'])
        src=str(info[0]['src'])
        myfile.write(str(desc)+"\t" +str(src) +"\r\n")
        filename, file_extension = os.path.splitext(src)
        url="https://network.satnogs.org" + src
        print(url)
        imgdat = requests.get(url, stream=True)
        if imgdat.status_code == 200:
            with open("c://tmp/q/"+ desc.replace('/','_').replace('|','_').replace(' ','_') +file_extension, 'wb') as f:
                imgdat.raw.decode_content = True
                shutil.copyfileobj(imgdat.raw, f)
            f.close()
# Close the file
myfile.close()