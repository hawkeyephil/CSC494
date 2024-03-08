from imdb import Cinemagoer 

ia = Cinemagoer() 

movie = ia.get_movie('0133093') 
print(movie) 

code = ia.get_by_name('Star Wars')

#info = ia.get_movie_infoset() 
#print(info) 

info = movie.get_current_info
print(info)

# print the genres of the movie
print('Genres:')
for genre in movie['genres']:
    print(genre) 

# Get the reviews for the movie
#print(ia.get_movie_reviews('0133093'))

ia.update(movie, ['reviews']) 
reviews = movie['reviews'] 

#print(reviews[50])

#for review in reviews:
    #print(f"Author: {review[1:20]}")
    #print(review[1:50])


#for review in movie['reviews']: 
    #print(review) 

