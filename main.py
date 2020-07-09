
import texthero as hero
import pandas as pd 
import matplotlib.pyplot as plt
import sys
from texthero import preprocessing

sys.path.append('/home/maheep/nlp/lib/python3.6/site-packages')
data = pd.read_csv('https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv')

print(data.head(10))

def preprocess(data):
  
  custom_pipeline = [preprocessing.fillna,
                     preprocessing.lowercase,
                     preprocessing.remove_digits,
                     preprocessing.remove_punctuation,
                     preprocessing.remove_diacritics,
                     preprocessing.remove_stopwords,
                     preprocessing.remove_whitespace,
                     preprocessing.stem
                    ]
  
  data['cleaned_text']  = data['text'].pipe(hero.clean,
                                                   custom_pipeline
                                                   )
  return data

data = preprocess(data)
print(data.head(10))

TW = hero.visualization.top_words(data['cleaned_text']).head(10)

plt.figure()
TW.plot.bar()
plt.show()

data['pca'] = (
               data['cleaned_text']
               .pipe(hero.clean)
               .pipe(hero.tfidf)
               .pipe(hero.pca)
              )

data['kmeans'] = (data['cleaned_text']
               .pipe(hero.tfidf)
               .pipe(hero.kmeans)
               )
   
data.head(10)

hero.scatterplot(data, col = 'pca', color = 'topic', title = 'BBC PCA')              

