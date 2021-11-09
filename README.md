# Data Science Portfolio
Repository detailing my portfolio of Data Science Projects

### [1. EDA on Supermarket Dataset](https://github.com/ankurdhamija83/Data-Science-Projects/tree/main/EDA/Supermarket)
The goal of this project is to do `Exploratory Data Analysis` on a Supermarket Dataset. The notebook contains following sections.

**a. Initial Data Exploration**
- Load the dataset
- Basic overview
- Check the shape of the dataset
- Display the list of columns
- Check for data types
- Set index
- Get summary statistics

``` bash
#Key commands used
pd.read_csv()
df.head()
df.shape
df.columns()
df.dtypes
df.set_index()
df.info()
df.describe()
```
---

**b. Univariate Analysis**
- Distribution of a numerical variable
- Histogram of multiple numerical variables
- Countplot

``` bash
#Key commands used
sns.distplot()
plt.axvline()
plt.legend()
df.hist()
sns.countplot()
```
---

**c. Bivariate Analysis**
- Scatterplot to understand relationship between numeric variables
- Regression plot
- Box plot
- Line plot

``` bash
#Key commands used
sns.scatterplot()
sns.regplot()
sns.boxplot()
sns.lineplot()
```
---

**d. Dealing with Duplicate rows and missing values**
- Check and remove Duplicate rows
- Impute missing values

``` bash
#Key commands used
df.isnull()
df.isnull().sum()
df.fillna()
```
---

**e. Correlation analysis**
- Check for correlation amongst numeric variables

``` bash
#Key commands used
np.corrcoef()
df.corr()
sns.heatmap()
np.round()
```
---



### [2. EDA on Movies Dataset](https://github.com/ankurdhamija83/Data-Science-Portfolio/tree/main/EDA/Movies)
The goal of this project is to do `Exploratory Data Analysis` on a Movies Dataset. The dataset contains a list of `44,691` movies and a total of `22` columns.

Columns contain information related to movie name, genre, actors, directors, franchise, revenue, budget and other related information.

We do an `Exploratory Data Analysis` using `Pandas` to help `answer following queries`


**a. Find the best and worst movies with**
- Highest revenue
- Lowest revenue
- Highest Profit (Revenue - Budget)
- Lowest Profit (Revenue - Budget)
- Highest Return on Investment (=Revenue / Budget) (only movies with Budget >= 10)
- Lowest Return on Investment (=Revenue / Budget) (only movies with Budget >= 10)
- Highest number of Votes
- Highest Rating (only movies with 10 or more Ratings)
- Lowest Rating (only movies with 10 or more Ratings)
- Highest Popularity


**b. Filter the dataset with following `complex queries`**
- Science Fiction Action Movie with Bruce Willis (sorted from high to low Rating)
- Movies with Uma Thurman and directed by Quentin Tarantino (sorted from short to long runtime)
- Most Successful Pixar Studio Movies between 2010 and 2015 (sorted from high to low Revenue)
- Action or Thriller Movie with original language English and minimum Rating of 7.5 (most recent movies first)

**c. Analyze the Dataset and find out whether Franchises (Movies that belong to a collection) are more successful than stand-alone movies**
- Mean revenue
- Median Return on Investment
- Mean budget raised
- Mean popularity
- Mean rating

**d. Find out most successful franchisees**
- Total number of movies
- Total & mean budget
- Total & mean revenue
- Mean rating

**e. Find out most successful directors**
- Total number of movies
- Total revenue
- Mean rating
