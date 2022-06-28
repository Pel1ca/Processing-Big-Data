import requests
import json
import pytube

#request ot get videos
url = r'https://youtube.googleapis.com/youtube/v3/search?part=id&q=giro%20italia&key=[KEY]'
response = requests.get(url) 
# print json content
data = json.loads(response.text)
videoIds = [vid['id']['videoId'] for vid in data['items']]
videoUrls = ["https://www.youtube.com/watch?v="+id for id in videoIds]

for i, url in enumerate(videoUrls):
    yt = pytube.YouTube(url)
    #stream = yt.streams.first()
    stream = yt.streams.get_highest_resolution()
    stream.download(r"/scrap/videos/.","video"+str(i)+".mp4")