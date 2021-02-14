# Building recommendation systems

Notes from [Datacamp's course](https://learn.datacamp.com/courses/building-recommendation-engines-in-python), [Eugene Yan's blog](https://eugeneyan.com/tag/recsys/) and elsewhere.

> Recommendation engines target a specific kind of machine learning problem, they are designed to suggest a product, service, or entity to a user based on other users, and their own feedback.

There is a many-to-many relationship between the items being recommended and the users. Users interact with many items. Each item is interacted with by many users.

- a better recommendation can be made for an item that has been given a lot of feedback
- more personalised recommendations can be given for a user that has given a lot of feedback (signal/interactions)

How a user's preferences are measured falls into two main groups; (1) implicit feedback and (2) explicit feedback;

<img src="md_refs/implicit_explicit.png">

## Non-personalised recommendations

Recommending items most commonly paired.

For example, if a user rates "The Great Gatsby" 5 stars, recommend the `most read` books of other users who also rated it 5 stars.

Alternatively, create a ranking system for books, for example `average rating`, or `average rating * customers read` or set a threshold in terms of popularity (read by x number); and provide the highest ranked books as recommendations.

```Python
# recommend what's most popular

# take the 50 most popular
movie_popularity = user_ratings_df["title"].value_counts()
popular_movies = movie_popularity[movie_popularity > 50].index
popular_movies_rankings =  user_ratings_df[user_ratings_df["title"].isin(popular_movies)]

# sort by their average rating
average_rating_df = popular_movies_rankings[["title", "rating"]].groupby('title').mean()
sorted_average_ratings = average_rating_df.sort_values(by="rating", ascending=False)
```

You can create pairs of items, the item most commonly seen with A. _Note_ we want both permutations, i.e. order matters;

<img src="md_refs/pairs.png" width=300>

```Python
import pandas as pd
from itertools import permutations
from typing import List

def create_pairs(x: List[str]) -> pd.DataFrame:
    pairs = pd.DataFrame(list(permutations(x.values, 2)),
        columns=["book_a", "book_b"]
        )

    return pairs

book_pairs = book_df.groupby("user_id")["book_title"].apply(create_pairs).reset_index(drop=True)
pair_counts = book_pairs.groupby(["book_a", "book_b"]).size()
pair_counts_df = pd.to_frame(name="size").reset_index()
pair_counts_df.sort_values(by="size", ascending=False, inplace=True)

# most common books after someone has read LotR
pair_counts_df[pair_counts_df.book_a == "Lord of the Rings"].head()
```

## Content filtering

Recommending items similar to items a user has liked in the past. We need a notion of similarity between items.

Content based recommendations, prioritises items with similar attributes. This allows you to recommend new items, as well as leveraging a long-tail of items. Content filtering works well when we have a lot of information about the items, but not much data on how people feel about them.

> This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it. [ref](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system)

Encode attributes as a vector to easily calculate distance and similarity between items;

<img src="md_refs/attributes.png" width=300>

```Python
"""
# convert
|| title || genre ||
| book a | genre a |
| book a | genre b |
| book b | genre a |
| book b | genre c |

# to 
|| title || genre a || genre b || genre c ||
| book a |      1   |  1       |     0    |
| book b |      1   |  0       |     0    |
"""

df = pd.crosstab(df_books.title, df_books.genre).reset_index()

df[df.index == "Lord of the Rings"]
```

### Calculating similarity

The `Jaccard Similarity` is the ratio of attributes two items have in common / by the total number of combined attributes

$\cap$ = intersection (overlap) between two arrays

$\cup$ = union (all elements) in two arrays

<img src="md_refs/jaccard.png">

```Python
# similarity between two items
from sklearn.metrics import jaccard_score

print(jaccard_score(
    np.asarray(df_book.loc["The Hobbit"].values), 
    np.asarray(df_book.loc["A Game of Thrones"].values)
))

# similarity between all items at once
from scipy.spatial_distance import pdist, squareform

#  pdist == pairwise distance
jaccard_distances = pdist(df_books.values, metric="jaccard")

# turns 1d array into nested-array
# subtract values from 1 as jaccard is a measure of difference
sq_jaccard_distances = 1 - squareform(jaccard_distances)

distance_df = pd.DataFrame(sq_jaccard_distances,
                            index=df_books.index,
                            columns=df_books.index)


def similarity(title1: str, title2: str, df: pd.DataFrame) -> int:
    """returns the similarity score of two books"""
    return df[title1][title2]
```

Resulting dataframe;

<img src="md_refs/jaccard_output.png">

`Cosine Similarity`

#### Cosine versus Jaccard

From Data Science Stack Exchange [ref](https://datascience.stackexchange.com/questions/5121/applications-and-differences-for-jaccard-similarity-and-cosine-similarity)

Jaccard Similarity;

$$s_{ij} = \frac{p}{p+q+r}$$

Where;

- $p$ = # of attributes positive for both objects (intersection)
- $q$ = # of attributes 1 for $i$ and 0 for $j$
- $r$ = # of attributes 0 for $i$ and 1 for $j$

Cosine Similarity;

$$\frac{A \cdot B}{\|A\|\|B\|}$$

${\|vector\|}$ - is a scalar, denoting the Euclidean norm of the vector (square root of the sum of squares)

> In cosine similarity, the number of common attributes is divided by the product of A and B's distance from zero. It's a measure of distance in high dimensions.


> Whereas in Jaccard Similarity, the number of common attributes is divided by the number of attributes that exists in at least one of the two objects.

In cosine similarity all values are between 0 and 1, where 1 is an exact match.

Cosine similarity is better for working with features that have more variation in their data, as opposed to attributes being boolean.

### Working without clear attributes

Often it will not have clear attribute labels relating to an item. However, if the item has text tied to it, we can use this description, or any text related to the item to create labels. This process is known as "Term Frequency Inverse Document Frequency" or TF-IDF to transform the text into something usable.

<img src="md_refs/tfidf.png">

By dividing the the count of word occurrences by total words in the document we reduce the impact of of common words.

```Python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metric.pairwise import cosine_similarity

# min_df -> only include words that occur at least twice
# max_df -> exclude words that occur in over 70% of descriptions
vectorizer = TfidfVectorizer(min_df=2, max_df=0.7)

vectorised_data = vectorizer.fit_transform(df_books.descriptions)

vectorised_data.to_array()  # row for each book, column for each feature

tfidf_df = pd.DataFrame(vectorised_data.to_array(),
                        columns=vectorizer.get_feature_names()
                        )
tfidf_df.index = df_books.book_titles  

# similarity between all rows
cosine_s_array = cosine_similarity(tfidf_df)

# plot a matrix book x book
cosine_similarity_df = pd.DataFrame(cosine_similarity_array, 
                            index=tfidf_df.index, 
                            columns=tfidf_df.index
                        )

# similarity between two rows
cosine_s = cosine_similarity(
                tfidf_df.loc["The Hobbit"].values.reshape(1, -1),
                tfidf_df.loc["Macbeth"].values.reshape(1, -1)
            )

# get similarities for a particular title
cosine_similarity_df.loc["Lord of the Rings"].sort_values(ascending=False)
```

### Presenting tastes in user profiles

Take all the items a user has engaged with and create mean scores across each attribute to create a "taste profile".

```Python
# list of title the user has read and enjoyed
user_books = df_user.books_enjoyed[df_user.user="1234"].values

books_enjoyed_df = tfidf_summary_df.reindex(user_books)  # df of books as rows, attributes as columns

user_tastes = books_enjoyed_df.mean()  # array of mean of column value

# to recommend, 
    # remove books already liked
tfidf_subset_df = tfidf_summary_df.drop(user_books, axis=0)

    # Calculate the cosine_similarity and wrap it in a DataFrame
    # columns is number of features
similarity_array = cosine_similarity(user_tastes.values.reshape(1, -1), tfidf_subset_df)

similarity_df = pd.DataFrame(similarity_array.T, 
                    index=tfidf_subset_df.index, 
                    columns=["similarity_score"]
                )
sorted_similarity_df = similarity_df.sort_values(
                            by="similarity_score",
                            ascending=False
                        )
```

## Collaborative filtering

Collaborative filtering finds users that have the most similar preferences to the user we are making recommendations for and based on that group's preferences, make suggestions.

> Collaborative filtering uses information on user behaviours, activities, or preferences to predict what other users will like based on item or user similarity. In contrast, content filtering is based solely on item metadata (i.e., brand, price, category, etc.). _-- Eugene Yan_

> This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts. [ref](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system)

We need to transform data into a matrix of users and the items they rated.

<img src="md_refs/collab_filtering1.png" width=400><br>

> Based on this matrix we can compare across users, here it is apparent that User_1 and User_3 have more similar preferences than User_1 and User_2.

### Handling sparse data

This matrix will be extremely sparse. Users won't have a expressed a positive or negative view towards the majority of items, you can't simply drop `NULLS` or fill in missing values with 0 (this is will impact our calculations).

One approach is to centre all the ratings around 0. 0 will therefore represent a neutral rating.

<img src="md_refs/collab_filtering2.png">

Do this by subtracting the user's mean rating from each score.

```Python
# make user_id the index, each column a move, each cell a rating
user_ratings_table = user_ratings.pivot(index="userId", columns="title", values="rating")

# each row represents a users mean rating
user_avg_rating = user_ratings_pivot.mean(axis=1)

# from each column, subtract the user's mean rating
user_ratings_pivot = user_ratings_pivot.sub(user_avg_rating, axis=0)
user_ratings_pivot.fillna(0)
```

These values should not be used for prediction. Only for comparing users.

Take the following example. Both users `B` and `C` are equally similar to user `A`. We cannot predict what the user will think of "The Matrix". If you filled `NULL` values with 0, you'd artificially make user `C` look more similar to user `A`.

<img src="md_refs/collab_filtering3.png" width=300>

### Item-based collaborative filtering

We can also find similarities between products simply by looking at the ratings they have received.

```Python
# transpose each user's rating of each film
# so that film titles are the index, user ratings the columns
move_ratings = user_ratings_table.transpose()
```

<img src="md_refs/collab_filtering4.png" width=300>

_Note: It appears that this does take into account who has rated the film, not simply the mean and distribution of ratings._ 

`cosine_similarity(sw_IV, sw_V) = [0.5357054]` despite the average rating between pretty different. Whereas Pulp Fiction and Star Wars 4 have a more similar rating on avarage, yet `cosine_similarity(sw_IV, pulp_fiction) = [-0.08386681]`.

```Python
from sklearn.metrics.pairwise import cosine_similarity

# with similarity scores centred around 0, cosine will be between -1 to 1
similarities = cosine_similarity(movie_ratings_centered)

cosine_similarity_df = pd.DataFrame(similarities, index=movie_ratings_centered.index, columns=movie_ratings_centered.index)

# Find the similarity values for a specific movie
cosine_similarity_series = cosine_similarity_df.loc['Star Wars: Episode IV - A New Hope (1977)'].sort_values(ascending=False)
```

### Using K-nearest neighbours

Predicting how a user might rate an item they have not yet seen. One approach is to find similar users using a K nearest neighbors model and see how they liked the item.

<img src="md_refs/k-nearest-neighbours.png>

`K-NN` finds the k users that are closest measured by a specified metric, to the user in question. It then averages the rating those users gave the item we are trying to get a rating for.

<img src="md_refs/user-user.png.png>

```Python
from sklearn.neighbors import KNeighborsRegressor

user_knn = KNeighborsRegressor(metric="cosine", n_neighbors=3)

# other_users_x = how users rated every other item in the catalogue
# other_users_y = how users rated the item in question
user_knn.fit(other_users_x, other_users_y)

# target_user_x = how the user in question has rated every item to date
user_user_pred = user_knn.predict(target_user_x)
user_user_pred  # now contains the predicted avg rating the user would give the item it has not seen
```

****

Ref: https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26
