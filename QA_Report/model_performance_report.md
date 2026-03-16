QA Report: Machine Learning Model Performance Evaluation

Objective: 

The objective of this report is to evaluate the performance of different machine learning models used for Titanic survival prediction on same data set.  

The analysis focuses on model testing from a QA perspective including:

- Confusion matrix evaluation
- Manual metric calculations
- Model comparison
- Feature importance analysis
- Final conclusion

Models Evaluated:

1. Logistic Regression
2. Random Forest Classifier


Results:
**Confusion Matrix Values of Logistic Regression**
| Actual \ Predicted | Not Survived | Survived |
| ------------------ | ------------ | -------- |
| Not Survived       | 98           | 12       |
| Survived           | 19           | 50       |

True Positive =  50
True Negative =  98
False Positive =  12
False Negative =  19

Total = TP + TN + FP + FN
Total = 50 + 98 + 12 + 19
Total = 179

Accuracy = (TP + TN) / Total
Accuracy = (50 + 98) / 179
Accuracy = 148 / 179
Accuracy = 0.826

**Accuracy ≈ 82.6%**

Precision = TP / (TP + FP)
Precision = 50 / (50 + 12)
Precision = 50 / 62
Precision = 0.806

**Precision ≈ 80.6%**


Recall = TP / (TP + FN)
Recall = 50 / (50 + 19)
Recall = 50 / 69
Recall = 0.724

**Recall ≈ 72.4%**

F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 = 2 × (0.806 × 0.724) / (0.806 + 0.724)

F1 = 2 × 0.583 / 1.53
F1 = 1.166 / 1.53
F1 = 0.762

**F1 Score ≈ 0.76**

Specificity = TN / (TN + FP)
Specificity = 98 / (98 + 12)
Specificity = 98 / 110
Specificity = 0.891

**Specificity ≈ 89.1%**

Conclusion:
The Logistic Regression model achieves an accuracy of 82.6%, indicating good overall predictive performance.
The precision score of 80.6% suggests that when the model predicts survival, it is correct most of the time.
However, the recall of 72.4% indicates that some passengers who actually survived were misclassified as not survived.
The specificity of 89.1% shows that the model is very effective at identifying passengers who did not survive.

**Feature Importance**
As per bar chart in evaluation section represent which feature model consider more important for good pridiction.
who_man
pclass
who_woman
who_child
sex_female

Conclusion:
The feature "who_man" has the highest coefficient magnitude, indicating that gender plays a significant role in survival prediction.
Passenger class (pclass) also strongly influences survival probability, as first-class passengers historically had better access to lifeboats.
Age and fare have relatively lower influence on the prediction outcome.





**Random Forest Confusion Matrix Values**
| Actual \ Predicted | Not Survived | Survived |
| ------------------ | ------------ | -------- |
| Not Survived       | 97           | 13       |
| Survived           | 20           |49        |


TN = 97
FP = 13
FN = 20
TP = 49

Total = TP + TN + FP + FN
Total = 49 + 97 + 13 + 20
Total = 179

Accuracy = (TP + TN) / Total
Accuracy = (49 + 97) / 179
Accuracy = 146 / 179
Accuracy = 0.815

**Accuracy ≈ 81.5%**

Precision = TP / (TP + FP)
Precision = 49 / (49 + 13)
Precision = 49 / 62
Precision = 0.790

**Precision ≈ 79.0%**

Recall = TP / (TP + FN)
Recall = 49 / (49 + 20)
Recall = 49 / 69
Recall = 0.710

**Recall ≈ 71.0%**

F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 = 2 × (0.79 × 0.71) / (0.79 + 0.71)

F1 = 2 × 0.561 / 1.50
F1 = 1.122 / 1.50
F1 = 0.748

**F1 Score ≈ 0.75**


Specificity = TN / (TN + FP)
Specificity = 97 / (97 + 13)
Specificity = 97 / 110
Specificity = 0.881

**Specificity ≈ 88.1%**


**Feature Importance (Random Forest image)**
fare
age
sex_male
who_man
sex_female

Conclusion:
The Random Forest model identifies fare and age as the most influential features.
Passengers who paid higher fares were more likely to survive, which is related
to higher passenger classes having better access to lifeboats.
Gender-related features such as sex_male and who_man also significantly influence
survival predictions, reflecting historical survival patterns of the Titanic disaster.



**Logistic Regression vs Random Forest**

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------- | ------------- |
| Accuracy  | 82.6%               | 81.5%         |
| Precision | 80.6%               | 79.0%         |
| Recall    | 72.4%               | 71.0%         |
| F1 Score  | 0.76                | 0.75          |

Conclusion:
Logistic Regression slightly outperforms Random Forest on this dataset
in terms of accuracy, precision, recall, and F1 score.

**Final QA Conclusion**

Both models demonstrate strong predictive performance on the Titanic dataset.
Logistic Regression slightly outperforms Random Forest in terms of
accuracy and F1 score.
Therefore, Logistic Regression is selected as the final model for
Titanic survival prediction in this project.



**Feature importance calculation method**

Logistic Regression
Importance = coefficient magnitude
Importance ∝ |coefficient|
Big coefficient = More Importance


Random Forest
Importance = how much feature reduces impurity
Feature importance ∝ reduction in decision tree impurity
Feature more used in tree split  = More Importance



Conclusion:
Different machine learning models may assign different importance to features even when trained on the same dataset. This occurs because each algorithm learns patterns using different mathematical approaches.
Logistic Regression evaluates feature importance based on the magnitude of model coefficients and assumes a linear relationship between features and the target variable.
Random Forest, on the other hand, determines feature importance based on how frequently a feature is used to split nodes in decision trees and how much it reduces impurity.
Because of these different learning mechanisms, the same dataset can produce different feature importance rankings across models.


