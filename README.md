# ML_Project51-ArticleRecommendationSystem

### Collaborative Filtering Recommendation System with ALS Matrix Factorization
This project implements a basic collaborative filtering recommendation system using the Alternating Least Squares (ALS) matrix factorization technique. It leverages user interaction data to recommend articles that users are likely to be interested in.

### Project Overview
### The code performs the following steps:

### Data Preparation:
#### 1.Imports necessary libraries:
```
pandas: Data manipulation and analysis
scipy.sparse: Working with sparse matrices
numpy: Numerical computations
random: Random number generation
implicit: Implicit matrix factorization library
```

#### 2.Loads the datasets:
```
shared_articles.csv: Contains information about articles
users_interactions.csv: Contains user interactions with articles (views, likes, etc.)
```

#### 3.Preprocesses the data:
```
Drops irrelevant columns from both datasets
Converts the eventType column in articles_df to categorical data
Creates a new DataFrame df by merging the user interactions and article information based on content ID
```

#### 4.Assigns weights to different interaction types:
```
Defines a dictionary event_type_strength that assigns weights to different interaction types (VIEW, LIKE, etc.)
Adds a new column eventStrength to df by applying the corresponding weight to each interaction type
```

#### 5.Reduces duplicates and groups data:
```
Removes duplicate entries from df
Groups the data by personId, contentId, and title and calculates the sum of eventStrength for each group
```

#### 6.Encodes categorical features:
```
Converts the categorical columns (personId, contentId, and title) to numerical representations using the astype method
```

#### 7.Creates sparse matrices:
```
Creates two sparse matrices:
sparse_content_person: This matrix represents the interaction strength between users and articles.
sparse_person_content: This matrix represents the interaction strength between articles and users (transposed version of sparse_content_person).
```

### Model Training:

#### 1.Initializes the ALS model:
```
Creates an instance of the AlternatingLeastSquares class from the implicit.als module.
Sets the number of latent factors (factors), regularization parameter (regularization), and number of iterations (iterations).
````

#### 2.Fits the model:
```
Trains the ALS model on the sparse_content_person matrix.
```

### Recommendation Generation:

#### 1.Defines the recommend function:

Takes a person ID, sparse person-content matrix, person and content vector matrices, and the desired number of recommendations as input.

Retrieves the interaction scores for the given person from the sparse matrix.

Adds 1 to all interaction scores to ensure articles with no interaction have a score of 1.

Sets scores of already interacted articles to 0.

Calculates the dot product of the person vector with all content vectors.

Scales the recommendation vector between 0 and 1 using MinMaxScaler.

Multiplies the scaled recommendation vector with the interaction vector to prioritize articles with no prior interaction.

Sorts the content indices based on the recommendation scores in descending order.

Retrieves titles and scores for the top num_contents recommended articles and returns them as a DataFrame.

#### 2.Generates recommendations for person ID 50:

Retrieves the trained person and content vector matrices.

Calls the recommend function with person ID 50 and other necessary parameters.

Prints the recommended articles and their scores.

#### 3.Verifies recommendations:

Shows the top 10 articles interacted with by person ID 50 for comparison.

#### 4.Generates recommendations for person ID 2:

Repeats the recommendation process for person ID 2 and prints the results.



### Getting Started
#### Prerequisites:
```
Python 3.x
Necessary libraries: pandas, scipy.sparse, numpy, random, implicit
```
#### Running the Code:
```
Download and unzip the project files.
Open a terminal or command prompt and navigate to the project directory.
Run the script: python collaborative_filtering.py
```

#### Output:
```
The script will print recommended articles with their scores for two example user IDs.
It will also show the top interacted articles for comparison.
```

#### Project Structure
```
collaborative_filtering.py: Main script containing data preparation, model training, and recommendation generation logic.
shared_articles.csv: Contains information about articles (title, category, etc.).
users_interactions.csv: Contains user interactions with articles (views, likes, etc.).
```

### Further Enhancements
This is a basic implementation, and several improvements can be made:

Different Matrix Factorization Techniques: Explore other algorithms like Singular Value Decomposition (SVD).

Hyperparameter Tuning: Optimize the number of latent factors, regularization parameter, and iterations for better performance.

Content-Based Features: Incorporate additional features like article content or user demographics for richer recommendations.

Evaluation Metrics: Implement metrics like precision, recall, and NDCG to evaluate the model's effectiveness.

This project provides a foundation for building a collaborative filtering recommendation system using ALS matrix factorization. Feel free to experiment and customize it further based on your specific needs and data characteristics.
