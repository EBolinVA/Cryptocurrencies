# Cryptocurrencies

## Overview
A fictitious investment bank, Accountability Accounting, is exploring a new cryptocurrency investment portfolio for its clients. The purpose of this project is to report on the cryptocurrencies on the trading market and group them into a classification system for the new investment. 

Unsupervised learning is utilized for this dataset in order to process the data to fit machine learning models. A clustering algorithm is used and visualizations are created.

## Resources
[Dataset ](https://github.com/EBolinVA/Cryptocurrencies/blob/main/Starter_Code/crypto_data.csv)

[View Code ](https://github.com/EBolinVA/Cryptocurrencies/blob/main/Starter_Code/crypto_clustering.ipynb)

Import Modules
``` 
# Initial imports
import pandas as pd
import hvplot.pandas
from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
```
## Preprocessing Data for Principal Component Analysis (PCA)

The original dataset includes the column "IsTrading" which tells us whether the cryptocurrency is currently being traded.
![image of original dataset]()

### Keep only currently trading cryptocurrencies
We want to include only cryptocurrencies that are currently active, so we keep only the records where ```"IsTrading" == True```

![image of code to keep only IsTrading ==True]()

### Keep only cryptocurrencies that have been mined
Next, after removing any rows with null values, we want to keep only the records of cryptocurrencies where coins have been mined. Here I used the ```.loc``` method to keep records with column "TotalCoinsMined" value greather than 0.

```no_null_df.loc[no_null_df['TotalCoinsMined'] >0]```

### Create a "CoinName" dataframe
Then after dropping the "IsTrading" column, we create a new dataframe that holds only the "CoinName" which we will come back to later. Next drop the "CoinName" from the original dataframe since it will not be used in the clustering algorithm.

### Use .get_dummies() to convert categorical data to indicator variables
We are left with four columns, two of which have string-type data, "Algorithm", and "ProofType". For these two columns, we use the ```.get_dummies()``` method to make the values numeric.

![image of dataframe with indicator values on all columns]()

### Standardize the data for machine learning 
```from sklearn.preprocessing import StandardScaler```
Now that all the columns have numeric values, the final preprocessing step is to standardize the data with StandardScaler from scikit-learn so they can be fit into machine learning models. This brings the values of columns to look more or less like standard normally distributed data.

![standardscaler code]()

## Reducing the Data Dimensions using PCA
```from sklearn.decomposition import PCA``` 

Instantiate PCA

```pca = PCA(n_components = 3)```

Previously, .get_dummies() took 2 columns with string-type data and created 96 columns with numeric indicator variables. The PCA process will now reduce a total of 98 columns down to 3 principal components which are just the three main dimensions of variation that contain most of the information in the original dataset. Reducing the dataset to 3 components from 98 variables allows the machine learning models to speed up the algorithms when the number of input features is too high.
 - Run the model on the data

    ```crypto_pca = pca.fit_transform(crypto_scaled)```

 - Create a new dataframe with the 3 principal components

![image of pca code]()

## Clustering Cryptocurrencies using K-means
```from sklearn.cluster import Kmeans```
- Find the best value of K using elbow curve. 

![image of elbow_curve]()

- Use the principal components data with K-means algorithm with a K value of 4, (n_clusters = 4), where the direction shifts in the curve.

![image of K-means model]()

## Visualizing Cryptocurrencies Results
- 3D Scatter with Clusters

![image of 3D scatter]()

- 2D Scatter with "TotalCoinsMined" and "TotalCoinSupply"

