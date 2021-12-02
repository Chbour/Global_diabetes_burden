# Overview over different files 

# contractions_def.py 
Dictionary of contraction defintions

# defines.py 
Definitions for Constants, Patterns and Grammar

# emotion_codes.py
Definitions and conversions for the emoticons and emojis

# preprocess.py
Following functions: 
- replace contractions (can't -> can not)
<<<<<<< HEAD
=======
- replace special words ("#type1", "t1d", "type one" -> TYPEONE)
>>>>>>> devAA
- replace hashtags (#diabetes -> diabetes), url's (https://protonmail.com/ -> URL), Users (@McKennan -> USER)
- tokenize ("I like diabetes research" -> "I", "like", "diabetes", "research")
- remove punctuations (ex. ;:,?"!)
- categorise emojis smile and emoticons :-) into categories like EMOT_SMILE
- all characters to lowercase
- replace numbers by its alphabetic writing ( 9 -> nine )
- remove stopwords (ex.: and, with, a, the)
- lemmatization (ex.: played -> play)
- stemming (ex.: reduce -> reduc)


# stopword_def.py
Definitions of stopword lists based on python's NLTK library