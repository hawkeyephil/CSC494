from imdb import IMDb

# create an instance of the IMDb class
ia = IMDb()

the_matrix = ia.get_movie('0133093') 
print(ia.get_by_name('Reacher', tv=False))
#print(sorted(the_matrix.keys()))

# show all information sets that can be fetched for a movie
#print(ia.get_movie_infoset()) #Information we can add. Keys will be added
ia.update(the_matrix, ['external reviews'])
ia.update(the_matrix, ['reviews'])
ia.update(the_matrix, ['critic reviews'])
# show which keys were added by the information set
#print(the_matrix.infoset2keys['external reviews']) #no external reviews, so no key is added
#print(the_matrix.infoset2keys['reviews']) # A lot of reviews. Adds key: 'reviews'
#print(the_matrix.infoset2keys['critic reviews']) #Adds the keys: 'metascore', and 'metacritic url'
# print(the_matrix['reviews'])
#print(sorted(the_matrix.keys())) #Check out the new keys that we have added 

reviews = the_matrix['reviews'] 

#print(reviews[20]) 