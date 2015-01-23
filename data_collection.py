from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.utils import import_simplejson
json = import_simplejson()


consumer_key = "ncMZ2CP7YmScHkLYwmfCYaTZz"
consumer_secret = "ZkFEJXxXEOUlqkhrJ14kzWakrXjqIe11de7ks28DyC79P31t9q"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token = "1157786504-XB3DXGrMmhvM1PAb6aeys3LJFYI9Y3LzS6veRHj"
access_token_secret = "8w69uDRm9PPA9iv3fNtkHPKP4FIq5SFtVbcE28wtcY5qx"

pos_emo = [':)', ':-)', ': )', ':D', '=)']
neg_emo = [':(', ':-(', ': (']
emo = pos_emo + neg_emo

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def on_data(self, raw_data):
        data = json.loads(raw_data)
	if data['retweet_count'] != 0 or data.has_key('retweeted_status') or data['lang'] != "en":
	    return True

	text = data['text']
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

	print(raw_data)
	print 
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=emo, languages='en')  # blocking!
