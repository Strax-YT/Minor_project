#given all the data regarding plotings
optimal_threshold = 0.600
accuracy = 0.901
roc_auc = 0.625

classification_report = {
    '0': {'precision': 0.36, 'recall': 0.30, 'f1-score': 0.33, 'support': 27},
    '1': {'precision': 0.94, 'recall': 0.95, 'f1-score': 0.95, 'support': 305},
    'accuracy': 0.90,
    'macro avg': {'precision': 0.65, 'recall': 0.63, 'f1-score': 0.64, 'support': 332},
    'weighted avg': {'precision': 0.89, 'recall': 0.90, 'f1-score': 0.90, 'support': 332}
}

confusion_matrix = np.array([[8, 19], [14, 291]])
#Features required in the dataset
feature_importances = pd.DataFrame({
    'feature': [
        'SerumElectrolytesSodium', 'BUN_Protein_Interaction', 'GFR', 'GFR_Creatinine_Ratio',
        'ProteinInUrine', 'GFR_Creatinine_Interaction', 'Kidney_Risk_Score', 'BUNLevels',
        'Protein_GFR_Ratio', 'HemoglobinLevels', 'Electrolyte_Hemoglobin_Ratio', 'SerumCreatinine',
        'BUN_Creatinine_Ratio'
    ],
    'importance': [3027, 2907, 2753, 2667, 2591, 2578, 2514, 2488, 2310, 1786, 1600, 1427, 1129]
})

# Plotting the data
plt.figure(figsize=(15, 10))

# 1. Optimal Threshold, Accuracy, and ROC-AUC
plt.subplot(2, 2, 1)
metrics = ['Optimal Threshold', 'Accuracy', 'ROC-AUC']
values = [optimal_threshold, accuracy, roc_auc]
sns.barplot(x=metrics, y=values, palette='viridis')
plt.title('Model Performance Metrics')
plt.ylabel('Value')

# 2. Classification Report
plt.subplot(2, 2, 2)
class_report_df = pd.DataFrame(classification_report).T
class_report_df.drop(columns=['support'], inplace=True)
class_report_df.plot(kind='bar', ax=plt.gca(), colormap='viridis')
plt.title('Classification Report')
plt.ylabel('Score')
plt.xticks(rotation=0)

# 3. Confusion Matrix
plt.subplot(2, 2, 3)
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 4. Feature Importances
plt.subplot(2, 2, 4)
sns.barplot(x='importance', y='feature', data=feature_importances.sort_values(by='importance', ascending=False), palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.tight_layout()
plt.show()
