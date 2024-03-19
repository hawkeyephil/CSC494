import tweepy

# Authenticate to Twitter
auth = tweepy.OAuthHandler("hSuKUtkQXJnCkbCjp5ojutooL", 
    "P9q6K9XsIIsSvMDQcPfSqEmjkrtftwg1Y0YIgEiToYLM8q3ETb")
auth.set_access_token("1765920341170200576-3so5fyUnhFw2PdVdHhgk1D0zU49G1B", 
    "i4vfPWLtmopXcG8DHd0LM70Ukgf3i4KtKZYb2XQwI3iwb")

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication") 

# Create API object
api = tweepy.API(auth) 

api.get_user("ElonMusk")

#for tweet in api.search_tweets(q="Python", lang="en", rpp=2):
    #print(tweet) 

