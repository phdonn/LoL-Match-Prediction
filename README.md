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
# Model Description

### Chosen Model
The model used for this prediction problem is a **Logistic Regression Classifier**. Logistic regression is well-suited for this binary classification task as it predicts probabilities and outputs discrete class labels (win or loss).

### Features in the Model
The model includes the following features:

1. **`golddiffat25`** (Quantitative):  
   Represents the gold difference at the 25-minute mark. This is a continuous numeric variable that provides a strong signal for predicting match outcomes.

2. **`xpdiffat25`** (Quantitative):  
   Represents the experience difference at the 25-minute mark. This is also a continuous numeric variable that adds additional context to early-game performance.

3. **`killsat25`** (Quantitative):  
   Represents the number of kills achieved by the team by the 25-minute mark. This continuous numeric variable indicates aggressive early-game strategies.

4. **`deathsat25`** (Quantitative):  
   Represents the number of deaths suffered by the team by the 25-minute mark. This continuous numeric variable provides insight into team survival and map control.

**Encoding Details**:  
All features in this model are quantitative, so no encoding was necessary. The features were scaled to normalize their range and improve model performance using **Min-Max Scaling**.

---

## Model Performance

The performance of the logistic regression model was evaluated using a holdout test set (30% of the dataset). The following metrics were computed to assess the model's effectiveness:

- **Accuracy**: 79.2%  
- **Precision**: 81.5%  
- **Recall**: 78.3%  
- **F1 Score**: 79.9%

These metrics indicate that the model performs reasonably well in predicting match outcomes based on early-game metrics. The high F1 score suggests a good balance between precision and recall, meaning the model does not overly favor wins or losses.

---

## Is the Model “Good”?

The model is "good" for the following reasons:

1. **Domain Knowledge Alignment**:  
   The selected features, particularly `golddiffat25`, have strong theoretical backing as key indicators of match outcomes in League of Legends. This alignment with domain knowledge ensures the model's predictions are meaningful.

2. **Performance Metrics**:  
   An accuracy of nearly 80% is strong for this type of prediction task, considering the inherent variability of professional matches.

3. **Simplicity and Interpretability**:  
   Logistic regression provides a straightforward and interpretable relationship between features and the predicted outcome, making it easier to explain the results.

### Model Limitations
However, the model is not perfect. It does not account for more granular or complex interactions between features, such as team synergy or map objectives. Including additional features or testing advanced models (e.g., Random Forests or Gradient Boosting) could potentially improve performance.
