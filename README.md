```markdown
# League of Legends Match Analysis

This project analyzes professional League of Legends match data to understand the factors influencing match outcomes. The dataset includes detailed game statistics, and the analysis focuses on the relationship between gold difference at 25 minutes (`golddiffat25`) and match outcomes (`result`).

## Data Cleaning Steps

The dataset originally contained 116,064 rows and 161 columns. The `golddiffat25` column, which represents the gold difference at the 25-minute mark, had 23,520 missing values, while the `result` column had no missing values. Rows with missing values in `golddiffat25` were dropped because this column is critical for analysis, reducing the dataset to 92,544 rows. Both `golddiffat25` and `result` columns were converted to numeric data types to ensure compatibility with analytical methods. The cleaned dataset now includes only complete data, ensuring the results are accurate and reliable.

The data cleaning process significantly impacted the analysis by removing incomplete data points that could have introduced errors. Although the dataset size was reduced, this step preserved the integrity of the results.

### Cleaned Data Head

```plaintext
   golddiffat25  result
180        1928.0       1
181        2943.0       1
182         660.0       1
183        5016.0       1
184        2194.0       1
```

## Univariate Analysis

A histogram was created to visualize the distribution of the `golddiffat25` column. The gold difference at 25 minutes shows a roughly normal distribution centered around zero, with most values falling between -5,000 and 5,000. This distribution suggests that most games are balanced around the 25-minute mark, with extreme gold advantages or disadvantages being relatively rare. This finding highlights the competitive nature of League of Legends matches and suggests that small gold advantages can significantly influence outcomes.

![Distribution of Gold Difference](images/distribution_plot.png)

## Bivariate Analysis

The relationship between `golddiffat25` and `result` was analyzed using a scatter plot and a box plot. The scatter plot demonstrated a moderate positive trend, indicating that higher gold differences are generally associated with winning outcomes. The box plot showed a clear distinction in the gold difference distributions for wins and losses. Teams that win matches (`result = 1`) tend to have positive gold differences, while teams that lose (`result = 0`) tend to have negative or neutral gold differences. This analysis confirms that early-game gold leads play a critical role in determining match outcomes.

![Gold Difference vs Match Outcome](images/bivariate_plot.png)

## Aggregate Statistics

The dataset was grouped by match outcome (`result`), and the mean gold difference for each outcome was calculated. Teams that won their matches had an average gold lead of 1,511.96, while losing teams had an average gold deficit of -1,511.96. This significant difference reinforces the idea that gold advantage at 25 minutes is a key indicator of success in League of Legends matches.

### Grouped Table

```plaintext
   result  golddiffat25
0       0  -1511.962828
1       1   1511.962828
```

## Missing Value Handling

Missing values in the `golddiffat25` column were handled by dropping rows where this column was `NaN`. This approach ensured that only complete and accurate data points were used in the analysis. Imputation was not performed because the gold difference is a key feature, and approximating its values could have introduced bias into the results.

## Conclusion

This analysis highlights the importance of early-game gold advantages in League of Legends. The findings suggest that teams with a higher gold difference at 25 minutes are significantly more likely to win. This insight provides valuable strategic implications for teams and fans alike, emphasizing the importance of early-game performance in professional play.
```
