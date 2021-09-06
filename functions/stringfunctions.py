import matplotlib.pyplot as plt
import io
import urllib, base64

def find_nth(big_string, search_string, n):
    start = big_string.find(search_string)
    while start >= 0 and n > 1:
        start = big_string.find(search_string, start+len(search_string))
        n -= 1
    return start

def get_image():
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    image = urllib.parse.quote(string)
    return image