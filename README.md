# Global diabetes burden

Based on https://github.com/WDDS/Tweet-Diabetes-Classification 

Our objective was to identify the regional differences of diabetes burden at the global level from the perspective of people with diabetes (PWD) to support the development of targeted diabetes programs that integrate the most relevant determinants of diabetes burden at a local level.

Link to the original article: 

Scripts can be found in Worldwide-Online-Diabetes-Observatory/jupyter_notebooks/

- Data collection : Script to collect Twitter data,
- Translation : Script to translate non-English tweets to English,
- Delete duplicates and RTs in mongodb : Requests to delete duplicates and Retweets directly in MongoDB,
- Delete similar tweets with cosine : Delete similar tweets based on cosine similarity,
- Training transformer jokes / Jokes classifier : Training and applying classifier to filter jokes and irony in tweets,
- Training transformer personal content / Personal content classifier : Training and applying classifier to filter out institutional tweets,
- Classifier Sex : Classifier to determine user's gender (Male, Female or Unknown),
- Classifier Type of diabetes : Classifier to determine user's type of diabetes (Type 1, Type 2 or Unknown),
- Emotions classification : Determines the probability of joy, anger, sadness and fear in each tweet,
- Geolocation : Geolocation process,
- fast_kmeans.py : K-means based on cosine,
- Silhouette analysis Kmeans : Silhouette analysis to determine the best K for Kmeans for each region, 
- Kmeans 7 regions : K-means applied to each of the seven World Bank Region, 
- Sentiment analysis with VADER : VADER Sentiment analysis (https://github.com/cjhutto/vaderSentiment)
