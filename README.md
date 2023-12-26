# Movie Recommendation System

## Overview

This repository implements a user-based collaborative filtering approach for movie recommendations using the MovieLens 100k dataset. The system finds similarities between users, predicts unrated movie ratings, and suggests the top 10 recommendations for a given user.

## User Similarity Calculation

The system employs two approaches to find user similarities: Pearson correlation and cosine similarity. Pearson correlation measures the linear relationship between users' ratings, while cosine similarity quantifies the cosine of the angle between user vectors, capturing their directional similarity.

## Finding Similar Users

The system identifies the top 10 closest users to a given user using both Pearson correlation and cosine similarity methods.

## Movie Rating Prediction

The `predict_movie_score` function predicts the rating for a particular movie (identified by `target_movie_id`) for the selected user (`selected_user`). It calculates the mean rating (`active_user_mean_rating`) for the selected user and then considers the weighted sum of ratings from similar users. The prediction is based on the similarity scores and the difference between the target movie's rating and the mean rating of each similar user.

### Prediction Formula:

predicted_score = active_user_mean_rating + weighted_sum/similarity_sum

## Cosine Similarity vs. Pearson Correlation

Cosine similarity is preferred over Pearson correlation for user-based collaborative filtering due to its ability to capture non-linear relationships and robustness to varying rating scales. Cosine similarity focuses on the directional similarity between users, making it more suitable for sparse and high-dimensional datasets.
