import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error, r2_score


# Initialize result dictionary
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "ROC-AUC Score": [],
    "Cross-Validation Score":[]
}

#------------------Loan Prediction Data Model Class-------------------------------------------------
class LoanPredictionModel:
    
    #columns: a list of column names to check for outliers.
    possibleOutliers = ['residential_assets_value', 'bank_asset_value','commercial_assets_value']
    
    #---------Load Loan Data set---------------------
    def loadData(self) :
        """
        Load data set from csv
        Returns
        -------
        data : DataFrame
            Data set as a DataFrame object.
        categorical_cols : List
            List of categorical columsn.
        numerical_cols : List
            List Of numerical columns.
        """
        data=pd.read_csv("D:\\Study\\ML\\Assignment\\loan_approval_dataset.csv")
        data.columns=data.columns.str.strip()
        # convert load status into numerical 0 or 1 for classification problem.
        data['loan_status']=data['loan_status'].apply( lambda x:1 if x==' Approved' else 0)

        categorical_cols =data.select_dtypes(include=['object']).columns.str.strip().tolist()
        numerical_cols   =data.select_dtypes(exclude=['object']).columns.str.strip().tolist()
        
        return data, categorical_cols, numerical_cols
    #-----------------------------------------------------------------
    
    #-------------Visualize the data against loan status--------------
    def visualize_data(self, data,  categorical_cols, numerical_cols):
        """
        VIsualize data as histogram for each feature against loan status.

        Parameters
        ----------
        data : DataFrame
            List of all data point.
        categorical_cols : List
            List of all categorical features.
        numerical_cols : List
            List of all numerical features.

        Returns
        -------
        None.

        """
        
        plt.figure(figsize=(15, 30))

        #---------------Plot all numerical features-------------
        for i, col in enumerate(numerical_cols):
            if col=='loan_status':
                continue
           
            # Distribution plot
            plt.subplot(10, 2, 2 * i + 2)
            sns.histplot(data,x=col, kde=True, hue=data['loan_status'])
            plt.xlabel(f"Distribution of {col}")
            plt.ylabel("Density")
            plt.title(f"Distribution Plot of {col}")
        
        # Apply tight layout for better spacing
        plt.tight_layout()
        plt.show()

        # ------------------Plot all categorical features----------
        plt.figure(figsize=(15, 5))
        for i,col in enumerate(categorical_cols):
            plt.subplot(1,3,i+1)
            sns.countplot(x=data[col],palette='Set2',hue=data['loan_status'])
            plt.title(f"Count plot of {col}")
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10,5))
        sns.heatmap(data[numerical_cols].corr(),annot=True,cmap='Reds')
        plt.show()

        cor=data[numerical_cols].corr()
        cor['loan_status'].abs().sort_values(ascending=False)

        loan=data.drop(['loan_id'], axis=1)

        sns.pairplot(loan)
        plt.show()
    #-----------------------------------------------------------------
    
    
    #--------------Data cleanup ---------------------------------
    def data_cleanup(self, data):
        """
        CLean data set and feature name,  
        and encode categorical data, clean unnecessary duplicate feature introduced for encoding.

        Parameters
        ----------
        data : DateFrame
            data set object.....

        Returns
        -------
        cleaned_data : DataFrame
            Cleaned dataset.

        """
        data.columns=data.columns.str.strip()

        # already corrected when data loaded at start.
        #data['loan_status']=data['loan_status'].apply( lambda x:1 if x==' Approved' else 0)
        
        
        cleaned_data = pd.get_dummies(data, dtype=int)
        cleaned_data.rename(columns = {'education_ Graduate':'education', 'self_employed_ Yes':'self_employed'}, inplace = True)
        #print(cleaned_data.columns)
        cleaned_data.drop(['education_ Not Graduate', 'self_employed_ No'], axis=1,inplace=True)

        return cleaned_data
    #-----------------------------------------------------------------

    def remove_outliers(self, data):
        """
        This function iterates over the given columns to identify outliers using the IQR method,
        and removes them from the DataFrame.
        
        param 
            data: The DataFrame containing the data.
        return
            cleaned_data: DataFrame with the outliers removed.
        """
        outlier_conditions = []

        # Iterate through each column and compute the outlier conditions
        for column in self.possibleOutliers:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_condition = (data[column] < lower_bound) | (data[column] > upper_bound)
            outlier_conditions.append(outlier_condition)
        
        # Combine all outlier conditions
        combined_outlier_condition = outlier_conditions[0]
        for condition in outlier_conditions[1:]:
            combined_outlier_condition |= condition  # Combine using OR condition
            
        # Remove outliers from the original DataFrame
        cleaned_data = data[~combined_outlier_condition]
        
        return cleaned_data
    #-----------------------------------------------------------------

    def print_confusion_matrix(self, title, y_test, y_pred):
        """
        Print Confusion matrix, test result set and predicted reultset.

        Parameters
        ----------
        title: title of the matrix for which the confusion matrix is generated.
        y_test : List of actual target values.
        y_pred : List of predictaed values.

        Returns
        -------
        None.

        """
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for : {title}')
        plt.show() 
    #-----------------------------------------------------------------

    # Function to append results
    def append_results(self, model_name, accuracy, precision, recall, f1_score, roc_auc_score,cross_score):
        """
        Cauptures results for later comparision 
        Parameters
        ----------
        model_name : Name of the model.
        accuracy : Model accurracy.
        precision : Model precision.
        recall : model recall.
        f1_score : model f1 score.
        roc_auc_score : model roc auc curve.
        cross_score : model cross score.

        Returns
        -------
        None.

        """
        results['Model'].append(model_name)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-Score'].append(f1_score)
        results['ROC-AUC Score'].append(roc_auc_score)
        results['Cross-Validation Score'].append(cross_score)
    #-----------------------------------------------------------------

    def evaluate(self, y_pred, y_test, model, X_train, y_train, y_pred_proba=None):
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ROC-AUC Score
        if y_pred_proba is not None:
            roc_auc =  roc_auc_score(y_test, y_pred_proba[:, 1])
        else :
            roc_auc = "N/A (no probabilities provided)"

        # Cross-validation score on the training set
        cross_score = cross_val_score(model, X_train, y_train, cv=5)
        
        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Cross-Validation Score: {cross_score.mean():.4f}")
        
        # Display predictions and actual values
        #display(pd.DataFrame(np.c_[y_pred, y_test], columns=["Prediction", "Actual"]))
        
        # Return metrics
        return accuracy, precision, recall, f1, roc_auc, cross_score.mean()
    #-----------------------------------------------------------------

    def plot_roc_curve(self, y_test, y_pred, auc_type):
        fpr, tpr, _ = roc_curve(y_test, y_pred[:,1])
        auc_score = roc_auc_score(y_test, y_pred[:,1])
        plt.plot(fpr, tpr, label=f'{auc_type} (AUC = {auc_score:.2f})')
    #-----------------------------------------------------------------

    def compare_roc_curve(self, y_test, y_pred_proba_lg, y_pred_proba_rf, y_pred_proba_dt, y_pred_proba_svc, y_pred_voting_proba):

        # Plot ROC-AUC curves
        plt.figure(figsize=(10, 8))

        # Logistic Regression
        self.plot_roc_curve(y_test, y_pred_proba_lg, 'Logistic Regression')

        # Random Forest
        self.plot_roc_curve(y_test, y_pred_proba_rf, 'Random Forest')

        # Decision Tree
        self.plot_roc_curve(y_test, y_pred_proba_dt, 'Decision Tree')

        # Support Vector Classifier
        self.plot_roc_curve(y_test, y_pred_proba_svc, 'SVC')

        # Voting Classifier
        self.plot_roc_curve(y_test, y_pred_voting_proba, 'Voting Classifier')

        # Plot settings
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC-AUC Curves for Multiple Classifiers')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    #-----------------------------------------------------------------


#------------- load data model ----------------------
model = LoanPredictionModel()
data, categorical_cols, numerical_cols = model.loadData()
print(data)
#----------------------------------------------

#--------visualize and data cleaning.......
model.visualize_data(data,  categorical_cols, numerical_cols)
cleaned_data = model.data_cleanup(data)
print(cleaned_data)
cleaned_data = model.remove_outliers(cleaned_data)


cleaned_data=cleaned_data.drop(['loan_id', 'self_employed', 'education', 'no_of_dependents' ], axis=1)


y = cleaned_data['loan_status']
X = cleaned_data.drop(['loan_status'], axis =1)

numerical_cols.remove('loan_id')
numerical_cols.remove('loan_status')
numerical_cols.remove('no_of_dependents')


plt.figure(figsize=(15,15))
print(cleaned_data)
sns.heatmap(cleaned_data.corr(),annot=True,cmap='coolwarm')
plt.show()

cor=cleaned_data.corr()
#print(cor)

cor['loan_status'].abs().sort_values(ascending=False)
#----------------------------------------------


#--------------- create Test and train data -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, random_state = 42)

#-------data Scaling -----------------
print(numerical_cols)
print("----------------------------")
scaler=StandardScaler()
scaler.fit(X_train[numerical_cols])
X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

dump(scaler, 'scaler.joblib')  # Save the scaler
#--------------------------------------

#----------------Decision Tree----------------------
lg=LogisticRegression()
lg.fit(X_train,y_train)
y_pred_lg=lg.predict(X_test)
y_pred_proba_lg=lg.predict_proba(X_test)
accuracy_lg, precision_lg, recall_lg, f1_lg, roc_auc_lg, cross_score_lg = model.evaluate(y_pred_lg, y_test, lg, X_train, y_train, y_pred_proba_lg)
model.append_results("Logistic Regression",accuracy_lg, precision_lg, recall_lg, f1_lg, roc_auc_lg, cross_score_lg)
model.print_confusion_matrix("Logistic Regression", y_test, y_pred_lg)
#---------------------------------------------------

#----------------Support vector Machine ---------------------
svc = SVC( probability=True,random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
y_pred_proba_svc = svc.predict_proba(X_test)
# Evaluate the SVC model
accuracy_svc, precision_svc, recall_svc, f1_svc, roc_auc_svc, cross_score_svc = model.evaluate(y_pred_svc, y_test, svc, X_train, y_train, y_pred_proba_svc)
# Append results for SVC
model.append_results("SVC", accuracy_svc, precision_svc, recall_svc, f1_svc, roc_auc_svc, cross_score_svc)
model.print_confusion_matrix("SVC", y_test, y_pred_svc)
#----------------------------------------------------------------

#----------------Decision Tree----------------------
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)
# Evaluate the best model
accuracy_dt, precision_dt, recall_dt, f1_dt, roc_auc_dt, cross_score_dt = model.evaluate(y_pred_dt, y_test, dt, X_train, y_train, y_pred_proba_dt)
# Append results for Decision Tree
model.append_results("Decision Tree", accuracy_dt, precision_dt, recall_dt, f1_dt, roc_auc_dt, cross_score_dt)
model.print_confusion_matrix("Decision Tree", y_test,y_pred_dt)

dt_importances=dt.feature_importances_
dt_importances_series = pd.Series(dt_importances,index=X_train.columns)
forest_importances_sorted=dt_importances_series.sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=forest_importances_sorted.index,y=forest_importances_sorted.values)
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.xticks(rotation=45)  # 
plt.show()
#----------------------------------------------------------------

#----------------Random Forest ----------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)
# Evaluate the RF model
accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf, cross_score_rf = model.evaluate(y_pred_rf, y_test, rf, X_train, y_train, y_pred_proba_rf)

# Append results for RF
model.append_results("Random Forest", accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf, cross_score_rf)
model.print_confusion_matrix("Random Forest", y_test,y_pred_rf)


rf_importances=rf.feature_importances_ 
forest_importances = pd.Series(rf_importances,index=X_train.columns)
forest_importances_sorted=forest_importances.sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=forest_importances_sorted.index,y=forest_importances_sorted.values)
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.xticks(rotation=45)  # 
plt.show()
#----------------------------------------------------------------


#----------------Soft vaoting classifier ----------------------
voting_clf= VotingClassifier(
    estimators=[
        ('lg', lg), 
        ('svc', svc),
        ('dt', dt), 
        ('rf', rf)
    ],
    voting='soft'   
)
voting_clf.fit(X_train,y_train)
y_pred_voting = voting_clf.predict(X_test)
y_pred_voting_proba = voting_clf.predict_proba(X_test)

# Evaluate the Voting Classifier
accuracy_voting, precision_voting, recall_voting, f1_voting, roc_auc_voting, cross_score_voting = model.evaluate(y_pred_voting, y_test, voting_clf, X_train, y_train, y_pred_voting_proba)

# Append results for Voting Classifier
model.append_results("Soft-Voting Classifier", accuracy_voting, precision_voting, recall_voting, f1_voting, roc_auc_voting, cross_score_voting)
model.print_confusion_matrix("Soft-Voting Classifier", y_test,y_pred_voting)
#----------------------------------------------------------------

print(X_test)
Models=pd.DataFrame(results)
Models.sort_values(by='Accuracy',ascending=False)
print(Models)

model.compare_roc_curve(y_test, y_pred_proba_lg, y_pred_proba_rf, y_pred_proba_dt, y_pred_proba_svc, y_pred_voting_proba)


dump(voting_clf, 'loan_prd_model.joblib')


#----------- Model to predict max loan amount that can be allowed----------------------------------------
# Define the target variable (loan_amount) and feature set (Xa)
ya = X_train['loan_amount']
Xa = X_train[['income_annum', 'loan_term', 'residential_assets_value', 
             'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]

# Split the data into training and testing sets (80% train, 20% test)
X_traina, X_testa, y_traina, y_testa = train_test_split(Xa, ya, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
loan_amount_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
loan_amount_model.fit(X_traina, y_traina)

# Train the Random Forest model
rf_model.fit(X_traina, y_traina)

# Predict the loan amount on the test data
predicted_loan_amount = loan_amount_model.predict(X_testa)
predicted_loan_amount_rf = rf_model.predict(X_testa)

# Print the predicted loan amounts for the test set
print("Predicted Loan Amounts:", predicted_loan_amount)

# Print the actual loan amounts from the test set for comparison
print("Actual Loan Amounts:", y_testa)

# Evaluate the model's performance
mse = mean_squared_error(y_testa, predicted_loan_amount)
r2 = r2_score(y_testa, predicted_loan_amount)

# Evaluate model performance for Random Forest
mse_rf = mean_squared_error(y_testa, predicted_loan_amount_rf)
r2_rf = r2_score(y_testa, predicted_loan_amount_rf)

# Print performance metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
print("\nRandom Forest Model:")
print("Mean Squared Error (MSE):", mse_rf)
print("R-squared (R²):", r2_rf)

# Compare models and determine which one is the best
if r2 > r2_rf:
    print("\nLinear Regression is the best model based on R-squared value.")
    dump(loan_amount_model, 'loan_amount_model.joblib')
else:
    print("\nRandom Forest is the best model based on R-squared value.")
    dump(rf_model, 'loan_amount_model.joblib')
# ------------------------------------------------------------------------------