from __future__ import absolute_import
import re
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.utils import import_simplejson
json = import_simplejson()

consumer_key = "ncMZ2CP7YmScHkLYwmfCYaTZz"
consumer_secret = "ZkFEJXxXEOUlqkhrJ14kzWakrXjqIe11de7ks28DyC79P31t9q"

access_token = "1157786504-XB3DXGrMmhvM1PAb6aeys3LJFYI9Y3LzS6veRHj"
access_token_secret = "8w69uDRm9PPA9iv3fNtkHPKP4FIq5SFtVbcE28wtcY5qx"

pos_emo = [":)", ":-)", ": )", ":D", "=)", ";)"]
neg_emo = [":(", ":-(", ": ("]
emo = pos_emo + neg_emo
class StdOutListener(StreamListener):

    def on_data(self, raw_data):
        data = json.loads(raw_data)
	if data['retweet_count'] != 0 or data.has_key('retweeted_status') or data['lang'] != "en":
	    return True

	text = data['text'].encode('ascii','ignore')
	pos = False
	neg = False
	if any(e in text for e in pos_emo):
            pos = True
	if any(e in text for e in neg_emo):
            neg = True
	
        emo_idx = None	
	if pos and not neg:
	    emo_idx = 4
	elif neg and not pos:
	    emo_idx = 0
	else:
	    return True 
	for e in emo:
            text = text.replace(e, "")

        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
	text = text.strip()

        if not text:
            return True 

        line = "\"" + str(emo_idx) + "\"," + "\"" + text + "\"\n"
        with open('data_collected.csv', 'a') as f:
	    f.write(line)
	
	print line
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=emo, languages='en')  # blocking!



