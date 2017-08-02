import pandas as pd
from scipy import misc
from sklearn.feature_extraction.text import CountVectorizer

# ORDINAL/NOMINAL CATEGORIZATION
df = pd.DataFrame({'mood': ['Mad', 'Happy', 'Unhappy', 'Neutral'],
                   'animal': ['Bird', 'Mammal', 'Fish', 'Bird']})
mood_order = ['Unhappy', 'Neutral', 'Happy']
df['mood_cat'] = df['mood'].astype('category',
                                   ordered=True,
                                   categories=mood_order).cat.codes
df['animal_cat'] = df['animal'].astype('category').cat.codes
df = pd.concat([df, pd.get_dummies(df['animal'])], axis=1)
print(df)

# TOKEN MATRIX
corpus = ['Authman ran faster than Harry because he is an athlete.',
          'Authman and Harry ran faster and faster.']
cv = CountVectorizer()
cv_matrix = cv.fit_transform(corpus).toarray()
print(cv.get_feature_names())
print(cv_matrix)

# READ IMAGE + RESIZE + FLATTEN
img = misc.imread('datasets/course_golden_ratio.png')[::2, ::2].flatten() / 255
print(img)
