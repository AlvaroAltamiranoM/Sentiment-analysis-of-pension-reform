# Sentiment Analysis of Brazil's 2019 Pension Reform

Natural language processing and supervised machine learning applied to public opinion on Brazil's 2019 pension reform (PEC 006/2019), using ~174,000 Portuguese-language tweets collected during the first congressional debate.

## Problem

Brazil's 2019 pension reform was consequential for the country's public finances, given its rapid population aging. This project measures public sentiment toward the reform in near real time, demonstrating an inexpensive alternative to traditional polling for tracking opinion on a specific policy debate.

## Data

- **~174,000 tweets** streamed via the Twitter API during the week of the first lower-chamber debate (July 9-15, 2019)
- Keyword query on Portuguese pension terms: *Previdencia*, *Aposentadoria*, *Aposentados*, *reformaprevidencia*, and variants
- Cleaned corpus of ~16 million words
- Two annotated training corpora for supervised classification: a ~9k-tweet set on Minas Gerais state government ("Minas") and a ~53k-tweet set on Brazilian TV programs ("USP"), each with ternary labels (negative / neutral / positive)

## Method

**NLP (spaCy Portuguese CNN model `pt_core_news_sm`, NLTK):**
- Part-of-speech tagging, syntactic dependency parsing, and named-entity recognition
- Concordance analysis of key terms (e.g., *idoso* / senior citizen)
- Geocoding of self-reported user locations via Nominatim / OpenStreetMap

**Text preprocessing:** HTML/XML stripping, removal of names/numbers/casing, Portuguese stopword removal, lemmatization, and word-level tokenization.

**Classification:** Multinomial Naive Bayes (selected for its token-independence assumption; random forests, SVM, and logistic regression tested as alternatives with comparable fit). Models trained on 75% of each annotated corpus and evaluated in both binary and ternary configurations.

## Results

- Tweet volume tracked the legislative timeline, peaking during debate and declining after the base text passed
- Sentiment classified as predominantly **negative**, consistent across both trained models
- Sentiment mapped by city for geo-located tweets

## Limitations

Twitter-based opinion measurement lacks the sampling rigor and external validation of household surveys or formal polls. Geo-location coverage is limited to users who share location. Results are illustrative of method rather than representative of the Brazilian population.

## Stack

`Python` | `spaCy` | `NLTK` | `scikit-learn` | `Twitter API` | `Nominatim`

## Write-up

Full methodology and visualizations: [Medium post](https://ajaltamiranomontoya.medium.com/twitter-sentiment-analysis-what-does-people-say-about-brazils-ongoing-pension-reform-830568e5b3fd)

---
*Note: a planned extension explored transfer learning with distilled BERT models for improved classification.*

