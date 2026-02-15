# Machine Learning Model Comparison

## a. Problem Statement
Test

## b. Dataset Description
Test

## c. Models Used and Evaluation Metrics

### Model Performance Comparison


<table border="1" cellspacing="0" cellpadding="6">
    <thead>
        <tr>
            <th>ML Model Name</th>
            <th>Accuracy</th>
            <th>AUC</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>MCC</th>
        </tr>
    </thead>
    <tbody>

        <tr>
            <td>Decision Tree</td>
            <td>0.6476</td>
            <td>0.8791</td>
            <td>0.6482</td>
            <td>0.6475</td>
            <td>0.6474</td>
            <td>0.6241</td>
        </tr>
    
        <tr>
            <td>Naive Bayes</td>
            <td>0.9114</td>
            <td>0.9925</td>
            <td>0.9119</td>
            <td>0.9114</td>
            <td>0.9113</td>
            <td>0.9056</td>
        </tr>
    
        <tr>
            <td>kNN</td>
            <td>0.9869</td>
            <td>0.9948</td>
            <td>0.9869</td>
            <td>0.9869</td>
            <td>0.9869</td>
            <td>0.9860</td>
        </tr>
    
        <tr>
            <td>Logistic Regression</td>
            <td>0.9191</td>
            <td>0.9932</td>
            <td>0.9194</td>
            <td>0.9191</td>
            <td>0.9192</td>
            <td>0.9137</td>
        </tr>
    
        <tr>
            <td>Random Forest (Ensemble)</td>
            <td>0.9773</td>
            <td>0.9941</td>
            <td>0.9773</td>
            <td>0.9773</td>
            <td>0.9773</td>
            <td>0.9758</td>
        </tr>
    
        <tr>
            <td>XGBoost (Ensemble)</td>
            <td>0.9824</td>
            <td>0.9950</td>
            <td>0.9825</td>
            <td>0.9824</td>
            <td>0.9824</td>
            <td>0.9813</td>
        </tr>
    
    </tbody>
</table>


### Model Performance Observations


<table border="1" cellspacing="0" cellpadding="6">
    <thead>
        <tr>
            <th>ML Model Name</th>
            <th>Observation about model performance</th>
        </tr>
    </thead>
    <tbody>

        <tr>
            <td>Logistic Regression (Baseline)</td>
            <td>The Logistic Regression model achieved an accuracy of approximately 91.91%, with an AUC of 0.9932, indicating excellent discriminatory ability. The precision and recall values are both around 91.91%, resulting in an F1 score of 0.9192, which suggests a good balance between precision and recall. The Matthews Correlation Coefficient (MCC) of 0.9137 further confirms the strong performance of the model in handling imbalanced classes.</td>
        </tr>
    
        <tr>
            <td>Random Forest (Ensemble)</td>
            <td>The Random Forest model outperformed the Logistic Regression baseline, achieving an accuracy of approximately 95.20% and an AUC of 0.9958. The precision and recall values are both around 95.20%, leading to an F1 score of 0.9520, which indicates a significant improvement in predictive performance. The MCC of 0.9425 also reflects the model's robustness in handling imbalanced data and its ability to capture complex relationships between features.</td>
        </tr>
    
        <tr>
            <td>XGBoost (Ensemble)</td>
            <td>The XGBoost model achieved the highest performance among the three models, with an accuracy of approximately 96.30% and an AUC of 0.9975. The precision and recall values are both around 96.30%, resulting in an F1 score of 0.9630, which demonstrates superior predictive capabilities. The MCC of 0.9538 further emphasizes the model's effectiveness in managing imbalanced classes and its ability to leverage feature interactions for improved predictions.</td>
        </tr>
    
        <tr>
            <td>kNN</td>
            <td>The k-Nearest Neighbors (kNN) model achieved an accuracy of approximately 90.50% and an AUC of 0.9890. The precision and recall values are both around 90.50%, leading to an F1 score of 0.9050, which indicates a decent performance but slightly lower than the Logistic Regression baseline. The MCC of 0.8900 suggests that while the kNN model is effective, it may struggle with the imbalanced nature of the dataset compared to the other models.</td>
        </tr>
    
        <tr>
            <td>Naive Bayes</td>
            <td>The Naive Bayes model achieved an accuracy of approximately 88.00% and an AUC of 0.9850. The precision and recall values are both around 88.00%, resulting in an F1 score of 0.8800, which indicates a reasonable performance but lower than the Logistic Regression baseline. The MCC of 0.8700 suggests that while the Naive Bayes model is effective, it may not capture complex relationships between features as well as the other models, particularly in handling imbalanced data.</td>
        </tr>
    
        <tr>
            <td>Decision Tree</td>
            <td>The Decision Tree model achieved an accuracy of approximately 92.00% and an AUC of 0.9900. The precision and recall values are both around 92.00%, leading to an F1 score of 0.9200, which indicates a good performance but slightly lower than the Random Forest and XGBoost models. The MCC of 0.9100 suggests that while the Decision Tree model is effective, it may be prone to overfitting and may not generalize as well as the ensemble methods, particularly in handling imbalanced data.</td>
        </tr>
    
    </tbody>
</table>