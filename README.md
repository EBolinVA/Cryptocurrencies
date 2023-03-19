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
Then after dropping the "IsTrading" column, we create a new dataframe that holds only the "CoinName" which we will come back to later. Then drop the "CoinName" from the original dataframe since it will not be used in the clustering algorithm.

### Use .get_dummies() to convert categorical data to indicator variables
We are left with four columns, two of which have string-type data, "Algorithm", and "ProofType". For these two columns, we use the ```.get_dummies()``` method to make the values numeric.

![image of dataframe with indicator values on all columns]()

### Standardize the data for machine learning 
Now that all the values are numeric, the data need to be standardized with StandardScaler from scikit-learn so they can be fit into machine learning models. This brings the values of columns to more or less look like standard normally distributed data.

![standardscaler code]()






