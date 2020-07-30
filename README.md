# simple-recommendation-algorithm-research

This is a graduation project.  
**Simple** implementation and research of some commonly used **recommendation algorithms**.  

## Installation dependencies

python      ...(v3.6.8)  
numpy       ...(vx.x.x)  
pandas      ...(vx.x.x)  
jieba       ...(vx.x.x)  
sklearn     ...(vx.x.x)  
matplotlib  ...(vx.x.x)  

## Data set description

name: MovieLens1M  
LINK：[https://grouplens.org/datasets/movielens/]  

## Project Tree

This is a project tree.

```
simple-recommendation-algorithm-research
├─ source                                              
│  ├─ movies.csv                                ...Movie dataset
│  └─ ratings.csv                               ...User rating dataset
├─ Analysis.py                                  ...Generate a line chart from tabular data
├─ Avg_global.py                                ...Calculate the global average RMSE
├─ Avg_item.py                                  ...Calculate the average RMSE of items rated by users
├─ Avg_user.py                                  ...Calculate the average RMSE of user ratings for items
├─ CB_create_item_profile.py                    ...Construct feature vector of item label
├─ CB_main.py                                   ...Content-based recommendation algorithm
├─ CF_create_user_profile.py                    ...Construct the feature vector of user interest
├─ CF_main.py                                   ...Collaborative filtering recommendation algorithm
├─ Fuction.py                                   ...Similarity algorithm and some other encapsulated functions
├─ MIX_algorithm_fusion.py                      ...Result-weighted mixed recommendation algorithm
├─ MIX_PROFILE.py                               ...Algorithm-fusion hybrid recommendation algorithm
├─ MIX_PROFILE_create_itemuser_profile.py       ...Construct a combination of item and user feature vectors
├─ MIX_result_weigh.py                          ...Feature-combined hybrid recommendation algorithm
├─ Package.py                                   ...The import package in all python files
├─ Split_train_test.py                          ...Split the original data set to generate training set and test set
└─ __main__.py                                  ...Main file, summarizing all recommended algorithms

```