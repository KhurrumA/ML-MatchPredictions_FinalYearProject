import sqlite3 #imports that are required 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,classification_report, precision_score, recall_score, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
#-----------------------------Checking if any class predictictions are missing
def check_missing_predictions(y_true, y_pred, label_encoder, model_name):#this function checks if the label was predicted by the model (was getting warnings with no draw in poisson as that makes sense)
    true_labels = set(y_true)
    predicted_labels = set(y_pred)
    missing = true_labels - predicted_labels
    if missing:
        missing_labels = label_encoder.inverse_transform(list(missing))
        print(f"[Warning] {model_name} did not predict the following class(es): {missing_labels}")
    else:
        print(f"{model_name} predicted all classes.")

#---------------First the data will be loaded and processed 
print(" Loading data") #data is loaded from the sqlite database that i created in the data scraping phase
conn = sqlite3.connect("matches.db")
df = pd.read_sql_query("SELECT * FROM matches", conn)#databse is read intoa  dataframe
conn.close()
print("Data has been loaded.")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")#Date is columns  is parced into actual datetime objects
df = df.dropna(subset=["Date"])
df = df[df["Date"] >= "2016-08-01"].sort_values("Date").reset_index(drop=True)#only matches after this data are kept as no statistical data for other matches prior
print("Dates parsed and filtered.")

#-------------------Need to calculate the form of a team(average points over last 5 matches)
def calculate_form_points(team, matches): #form and head to head matches are calculated
    points = 0
    for i, row in matches.iterrows():
        if team == row["Home Team"]:
            points += 3 if row["Home Goals"] > row["Away Goals"] else 1 if row["Home Goals"] == row["Away Goals"] else 0
        elif team == row["Away Team"]:
            points += 3 if row["Away Goals"] > row["Home Goals"] else 1 if row["Away Goals"] == row["Home Goals"] else 0
    return points / len(matches) if len(matches) > 0 else 0

home_form_scores, away_form_scores = [], []
h2h_home_wins, h2h_away_wins, h2h_draws = [], [], [] #empty lists to store stats that we calculate

for idx, row in df.iterrows(): #goes through every match in the dataset
    match_date = row["Date"]
    home_team = row["Home Team"]
    away_team = row["Away Team"]
    past_matches = df[df["Date"] < match_date]#current matche's date and teams

    last_home = past_matches[(past_matches["Home Team"] == home_team) | (past_matches["Away Team"] == home_team)].tail(5)
    last_away = past_matches[(past_matches["Home Team"] == away_team) | (past_matches["Away Team"] == away_team)].tail(5)
    home_form_scores.append(calculate_form_points(home_team, last_home))
    away_form_scores.append(calculate_form_points(away_team, last_away))#calculates and stores scores for form

#---------------------------------Calculate Head to Head matches   

    h2h = past_matches[((past_matches["Home Team"] == home_team) & (past_matches["Away Team"] == away_team)) | #gets matches where both teams played each other
                       ((past_matches["Home Team"] == away_team) & (past_matches["Away Team"] == home_team))]

    hw, aw, d = 0, 0, 0 #win for both teams and draw counters
    for i, h2h_row in h2h.iterrows(): #loops through every head to head match
        h = h2h_row["Home Goals"]
        a = h2h_row["Away Goals"]
        
        if h2h_row["Home Team"] == home_team:
            if h > a:
                hw = hw + 1
            elif a > h:
                aw = aw + 1
            else:
                d = d + 1
        else:
            if a > h:
                hw += 1
            elif h > a:
                aw = aw + 1
            else:
                d = d + 1 #calulates win or draw

    h2h_home_wins.append(hw) #appends the lists
    h2h_away_wins.append(aw)
    h2h_draws.append(d)

df["home_form_score"] = home_form_scores #all of these stats are then added to the dataframe
df["away_form_score"] = away_form_scores
df["h2h_home_wins"] = h2h_home_wins
df["h2h_away_wins"] = h2h_away_wins
df["h2h_draws"] = h2h_draws

#------------------cleaning the numeric columns
numeric_cols = [
    "Home xG", "Away xG",
    "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession",
    "home_passing_accuracy", "away_passing_accuracy",
    "home_corners", "away_corners",
    "home_touches", "away_touches"
]#list of all column names thaat have numerical data-performance related stats

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")#converts column to float for each column listed. coerce just used to convert data that is not numerical to NaN. NaN represents missing data

df["xg_diff"] = df["Home xG"] - df["Away xG"]
df["form_diff"] = df["home_form_score"] - df["away_form_score"]
df["possession_diff"] = df["home_possession"] - df["away_possession"]#more features engineered

#----------------------------Preparing the training data
features = numeric_cols + [
    "home_form_score", "away_form_score",
    "xg_diff", "form_diff", "possession_diff",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws"
]#more features engineered and added in 

df_model = df.dropna(subset=features + ["Result"])#dataframe filtered in rows that have no missing values in any of the target label(Result)
X = df_model[features]#feature matrix created (X) by selecting only the columns defined in features. This is waht the model will learn from 
y = df_model["Result"]#target vetor "y". This is what we want the model to predict

le = LabelEncoder()#label encoder converts strings in label y to values. E.g winn loss draw given numbers so model can interpret them
y_encoded = le.fit_transform(y)#transforms y (match result) into encoded integers

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)#random state is 42 as it gives best result after trying many different numbers.was only a difference of 0.005 give or take  
#data split into training-80 and evaluation-20

#-------------------------Training the Random Forest Model
rf_model = RandomForestClassifier(#creating RF model
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)#training the model
rf_preds = rf_model.predict(X_test)#predicting on the test set
rf_accuracy = accuracy_score(y_test, rf_preds)#this is the accuracy
rf_f1 = f1_score(y_test, rf_preds, average="macro")#this is the f1 score
rf_precision = precision_score(y_test, rf_preds, average='macro')#this is the precision
rf_recall = recall_score(y_test, rf_preds, average='macro')#this is recall
check_missing_predictions(y_test, rf_preds, le, "Random Forest")#checks to see if a class is missing

#--------------------------Training the XGBoost model
xgb_model = XGBClassifier(#Creating XGB model
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1,
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)#same as rf
xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
xgb_precision = precision_score(y_test, xgb_preds, average='macro')
xgb_recall = recall_score(y_test, xgb_preds, average='macro')
check_missing_predictions(y_test, xgb_preds, le, "XGBoost")

# My own take on Logistic Regression by using more features------------------
lr_full_model = Pipeline([#creating model
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        penalty='l1',
        C=10,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    ))
])


lr_full_model.fit(X_train, y_train)#training model

lr_full_preds = lr_full_model.predict(X_test)
lr_full_accuracy = accuracy_score(y_test, lr_full_preds)
lr_full_f1 = f1_score(y_test, lr_full_preds, average='macro')
lr_full_precision = precision_score(y_test, lr_full_preds, average='macro')
lr_full_recall = recall_score(y_test, lr_full_preds, average='macro')#evaluation parametes

check_missing_predictions(y_test, lr_full_preds, le, "Logistic Regression (All Features)")

# --------------------------Combining XGBoost, Random Forest and updated Logistic Regression into an ensemble
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('lr', lr_full_model)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)#training the model
ensemble_preds = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
ensemble_f1 = f1_score(y_test, ensemble_preds, average='macro')
ensemble_precision = precision_score(y_test, ensemble_preds, average='macro')#evaluation parameters
ensemble_recall = recall_score(y_test, ensemble_preds, average='macro')
check_missing_predictions(y_test, ensemble_preds, le, "Ensemble")


# ------------------------Poisson Model (Loukas et al., 2024) 
df_poisson = df[["Date", "Home Team", "Away Team", "Home Goals", "Away Goals"]].dropna()#used the same formula in the paper to get this model as there was no code. will be referenced in the dissertaion
mu = df_poisson["Away Goals"].mean()#overall average away goals
mu_home = np.log(df_poisson["Home Goals"].mean()) - np.log(mu)#home advantage

teams = pd.concat([df_poisson["Home Team"], df_poisson["Away Team"]]).unique()#store strength values
attack_strength, defense_strength = {}, {}

for team in teams:#calculate attack and defence stregths for all teams
    att_goals = df_poisson[df_poisson["Home Team"] == team]["Home Goals"].sum() + \
                df_poisson[df_poisson["Away Team"] == team]["Away Goals"].sum()
    def_goals = df_poisson[df_poisson["Home Team"] == team]["Away Goals"].sum() + \
                df_poisson[df_poisson["Away Team"] == team]["Home Goals"].sum()
    matches = len(df_poisson[(df_poisson["Home Team"] == team) | (df_poisson["Away Team"] == team)])
    attack_strength[team] = np.log(att_goals / matches + 1e-5) - np.log(mu)
    defense_strength[team] = np.log(def_goals / matches + 1e-5) - np.log(mu)

def predict_lambda(home_team, away_team):#predict expectted goals for both teams
    att_home = attack_strength.get(home_team, 0)
    def_away = defense_strength.get(away_team, 0)
    att_away = attack_strength.get(away_team, 0)
    def_home = defense_strength.get(home_team, 0)
    lambda_home = np.exp(np.log(mu) + mu_home + att_home + def_away)
    lambda_away = np.exp(np.log(mu) + att_away + def_home)
    return lambda_home, lambda_away

def simulate_result(lh, la, n=1000):#simulate match outcomes
    h_goals = poisson.rvs(lh, size=n)
    a_goals = poisson.rvs(la, size=n)
    outcomes = ["Home Win" if h > a else "Away Win" if a > h else "Draw" for h, a in zip(h_goals, a_goals)]
    return max(set(outcomes), key=outcomes.count)

true_results = []
pred_results = []

for i, row in df_poisson.iterrows():#simulate all matches in dataset
    actual = "Home Win" if row["Home Goals"] > row["Away Goals"] else \
             "Away Win" if row["Away Goals"] > row["Home Goals"] else "Draw"
    true_results.append(actual)
    lh, la = predict_lambda(row["Home Team"], row["Away Team"])
    pred_results.append(simulate_result(lh, la))

poisson_accuracy = accuracy_score(true_results, pred_results)#evaluation of model
poisson_f1 = f1_score(true_results, pred_results, average="macro")

le_poisson = LabelEncoder()
y_true_poisson = le_poisson.fit_transform(true_results)
y_pred_poisson = le_poisson.transform(pred_results)

poisson_precision = precision_score(y_true_poisson, y_pred_poisson, average='macro')
poisson_recall = recall_score(y_true_poisson, y_pred_poisson, average='macro')
check_missing_predictions(y_true_poisson, y_pred_poisson, le_poisson, "Poisson")



# ----------------Training Logistic Regression only uses basic features--------------------
lr_features = [
    "Home xG", "Away xG",
    "home_possession"
]

X_lr = df_model[lr_features]
y_lr = df_model["Result"]
y_lr_encoded = le.fit_transform(y_lr)

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr_encoded, test_size=0.2, random_state=42)

lr_model = LogisticRegression(
    penalty='l1',
    C=10,
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_lr, y_train_lr)

lr_preds = lr_model.predict(X_test_lr)
lr_accuracy = accuracy_score(y_test_lr, lr_preds)
lr_f1 = f1_score(y_test_lr, lr_preds, average='macro')
lr_precision = precision_score(y_test_lr, lr_preds, average='macro')#mostly same but less features as my own
lr_recall = recall_score(y_test_lr, lr_preds, average='macro')
check_missing_predictions(y_test_lr, lr_preds, le, "Logistic Regression")

#----------------------------------KNN from paper referenced in dissertation


knn_features = [#using basic features
    "home_form_score", "away_form_score",
    "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession"
]


X_knn = df_model[knn_features].dropna()#drop rows with missing values in these features
y_knn = df_model.loc[X_knn.index, "Result"]


le_knn = LabelEncoder()#encode the target
y_knn_encoded = le_knn.fit_transform(y_knn)


X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(#train test split
    X_knn, y_knn_encoded, test_size=0.2, random_state=42, stratify=y_knn_encoded
)


knn_model = KNeighborsClassifier(n_neighbors=5)#train knn model
knn_model.fit(X_train_knn, y_train_knn)


y_pred_knn = knn_model.predict(X_test_knn)#evaluation
knn_acc = accuracy_score(y_test_knn, y_pred_knn)
f1_macroKNN = f1_score(y_test_knn, y_pred_knn, average="macro")
knn_precision = precision_score(y_test_knn, y_pred_knn, average='macro')
knn_recall = recall_score(y_test_knn, y_pred_knn, average='macro')
check_missing_predictions(y_test_knn, y_pred_knn, le_knn, "KNN")



# -----------------------------Results
print("\nMODEL PERFORMANCE SUMMARY")
print("───────────────────────────────────────────")
print("Random Forest:")
print(f"   Accuracy : {rf_accuracy:.4f}")
print(f"   F1 Score : {rf_f1:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall   : {rf_recall:.4f}")
print("───────────────────────────────────────────")
print("XGBoost:")
print(f"   Accuracy : {xgb_accuracy:.4f}")
print(f"   F1 Score : {xgb_f1:.4f}")
print(f"   Precision: {xgb_precision:.4f}")
print(f"   Recall   : {xgb_recall:.4f}")
print("───────────────────────────────────────")
print("Ensemble (RF + XGBoost):")
print(f"   Accuracy : {ensemble_accuracy:.4f}")
print(f"   F1 Score : {ensemble_f1:.4f}")
print(f"   Precision: {ensemble_precision:.4f}")
print(f"   Recall   : {ensemble_recall:.4f}")
print("──────────────────────────────────────────")
print("Poisson Model:")
print(f"   Accuracy : {poisson_accuracy:.4f}")
print(f"   F1 Score : {poisson_f1:.4f}")
print(f"   Precision: {poisson_precision:.4f}")
print(f"   Recall   : {poisson_recall:.4f}")
print("──────────────────────────────────────────")
print("Logistic Regression with full features:")
print(f"   Accuracy : {lr_full_accuracy:.4f}")
print(f"   F1 Score : {lr_full_f1:.4f}")
print(f"   Precision: {lr_full_precision:.4f}")
print(f"   Recall   : {lr_full_recall:.4f}")
print("─────────────────────────────────────────")
print("Logistic Regression Simpe:")
print(f"   Accuracy : {lr_accuracy:.4f}")
print(f"   F1 Score : {lr_f1:.4f}")
print(f"   Precision: {lr_precision:.4f}")
print(f"   Recall   : {lr_recall:.4f}")
print("────────────────────────────────────────────")
print("k-Nearest Neighbours (KNN):")
print(f"   Accuracy : {knn_acc:.4f}")
print(f"   F1 Score : {f1_macroKNN:.4f}")
print(f"   Precision: {knn_precision:.4f}")
print(f"   Recall   : {knn_recall:.4f}")
print("────────────────────────────────────────")
#--------------------------------------------------------Coss Validation score
print("Cross Validation Results:")
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1_macro',
    'precision': 'precision_macro',
    'recall': 'recall_macro'
}

# Random Forest
rf_results = cross_validate(rf_model, X, y_encoded, cv=5, scoring=scoring)
print("Random Forest:")
print(f"   Accuracy : {rf_results['test_accuracy'].mean():.4f}")
print(f"   F1 Score : {rf_results['test_f1'].mean():.4f}")
print(f"   Precision: {rf_results['test_precision'].mean():.4f}")
print(f"   Recall   : {rf_results['test_recall'].mean():.4f}")


# XGBoost
xgb_results = cross_validate(xgb_model, X, y_encoded, cv=5, scoring=scoring)
print("XGBoost:")
print(f"   Accuracy : {xgb_results['test_accuracy'].mean():.4f}")
print(f"   F1 Score : {xgb_results['test_f1'].mean():.4f}")
print(f"   Precision: {xgb_results['test_precision'].mean():.4f}")
print(f"   Recall   : {xgb_results['test_recall'].mean():.4f}")

#New Logistic Regression using all features

lr_full_results = cross_validate(lr_full_model, X, y_encoded, cv=5, scoring=scoring)

# Output results
print("Logistic Regression (All Features):")
print(f"   Accuracy : {lr_full_results['test_accuracy'].mean():.4f}")
print(f"   F1 Score : {lr_full_results['test_f1'].mean():.4f}")
print(f"   Precision: {lr_full_results['test_precision'].mean():.4f}")
print(f"   Recall   : {lr_full_results['test_recall'].mean():.4f}")

# Logistic Regression simple
lr_results = cross_validate(lr_model, X_lr, y_lr_encoded, cv=5, scoring=scoring)
print("Logistic Regression Simple:")
print(f"   Accuracy : {lr_results['test_accuracy'].mean():.4f}")
print(f"   F1 Score : {lr_results['test_f1'].mean():.4f}")
print(f"   Precision: {lr_results['test_precision'].mean():.4f}")
print(f"   Recall   : {lr_results['test_recall'].mean():.4f}")

# Ensemble
ensemble_results = cross_validate(ensemble_model, X, y_encoded, cv=5, scoring=scoring)
print("Ensemble (RF + XGBoost + Updated LR):")
print(f"   Accuracy : {ensemble_results['test_accuracy'].mean():.4f}")
print(f"   F1 Score : {ensemble_results['test_f1'].mean():.4f}")
print(f"   Precision: {ensemble_results['test_precision'].mean():.4f}")
print(f"   Recall   : {ensemble_results['test_recall'].mean():.4f}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
poisson_acc_list = []
poisson_f1_list = []
poisson_prec_list = []
poisson_recall_list = []

true_results = np.array(true_results)
pred_results = np.array(pred_results)

le_poisson = LabelEncoder()
y_true_enc = le_poisson.fit_transform(true_results)
y_pred_enc = le_poisson.transform(pred_results)

for train_index, test_index in kf.split(y_true_enc): #splits used to evaluate poisson prediction
    y_true_fold = y_true_enc[test_index]
    y_pred_fold = y_pred_enc[test_index]
    poisson_acc_list.append(accuracy_score(y_true_fold, y_pred_fold))
    poisson_f1_list.append(f1_score(y_true_fold, y_pred_fold, average='macro'))
    poisson_prec_list.append(precision_score(y_true_fold, y_pred_fold, average='macro'))
    poisson_recall_list.append(recall_score(y_true_fold, y_pred_fold, average='macro'))

print("Poisson Model:")
print(f"   Accuracy : {np.mean(poisson_acc_list):.4f}")
print(f"   F1 Score : {np.mean(poisson_f1_list):.4f}")
print(f"   Precision: {np.mean(poisson_prec_list):.4f}")
print(f"   Recall   : {np.mean(poisson_recall_list):.4f}")


knn_results = cross_validate(knn_model, X_knn, y_knn_encoded, cv=5, scoring=scoring)
print("k-Nearest Neighbours (KNN):")
print(f"   Accuracy : {knn_results['test_accuracy'].mean():.4f}")
print(f"   F1 Score : {knn_results['test_f1'].mean():.4f}")
print(f"   Precision: {knn_results['test_precision'].mean():.4f}")
print(f"   Recall   : {knn_results['test_recall'].mean():.4f}")


#----------------------------------------Chciking if any class is missing a prediction in all of the models
check_missing_predictions(y_test, rf_preds, le, "Random Forest")
check_missing_predictions(y_test, xgb_preds, le, "XGBoost")
check_missing_predictions(y_test, lr_full_preds, le, "Logistic Regression (All Features)") 
check_missing_predictions(y_test, ensemble_preds, le, "Ensemble")
check_missing_predictions(y_test_lr, lr_preds, le, "Logistic Regression")
check_missing_predictions(y_test_knn, y_pred_knn, le_knn, "KNN")
check_missing_predictions(y_true_poisson, y_pred_poisson, le_poisson, "Poisson")
models = ['Random Forest', 'XGBoost', 'Logistic Regression (All Features)', 'Ensemble', 'Poisson', 'Logistic Regression', 'KNN']
#------------------------------------------plotting the scores
models_cv = ['Random Forest', 'XGBoost', 'Logistic Regression (All Features)', 'Logistic Regression', 'Ensemble', 'KNN', 'Poisson']

accuracies_cv = [
    rf_results['test_accuracy'].mean(),
    xgb_results['test_accuracy'].mean(),
    lr_full_results['test_accuracy'].mean(),
    lr_results['test_accuracy'].mean(),
    ensemble_results['test_accuracy'].mean(),
    knn_results['test_accuracy'].mean(),
    np.mean(poisson_acc_list)
]

f1_scores_cv = [
    rf_results['test_f1'].mean(),
    xgb_results['test_f1'].mean(),
    lr_full_results['test_f1'].mean(),
    lr_results['test_f1'].mean(),
    ensemble_results['test_f1'].mean(),
    knn_results['test_f1'].mean(),
    np.mean(poisson_f1_list)
]

precisions_cv = [
    rf_results['test_precision'].mean(),
    xgb_results['test_precision'].mean(),
    lr_full_results['test_precision'].mean(),
    lr_results['test_precision'].mean(),
    ensemble_results['test_precision'].mean(),
    knn_results['test_precision'].mean(),
    np.mean(poisson_prec_list)
]

recalls_cv = [
    rf_results['test_recall'].mean(),
    xgb_results['test_recall'].mean(),
    lr_full_results['test_recall'].mean(),
    lr_results['test_recall'].mean(),
    ensemble_results['test_recall'].mean(),
    knn_results['test_recall'].mean(),
    np.mean(poisson_recall_list)
]
def plot_metric(values, title, ylabel):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models_cv, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.xticks(rotation=15)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

plot_metric(accuracies_cv, "Cross-Validated Model Comparison - Accuracy", "Accuracy")
plot_metric(f1_scores_cv, "Cross-Validated Model Comparison - F1 Score", "F1 Score")
plot_metric(precisions_cv, "Cross-Validated Model Comparison - Precision", "Precision")
plot_metric(recalls_cv, "Cross-Validated Model Comparison - Recall", "Recall")


# -------------------------------------Confusion Matrix Function
def show_confusion_matrix(y_true, y_pred, encoder, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"{title} - Confusion Matrix")
    plt.xticks(rotation=15)
    plt.show()

# ------------------------------Display confusion matrices for all models

show_confusion_matrix(y_test, rf_preds, le, "Random Forest")
show_confusion_matrix(y_test, xgb_preds, le, "XGBoost")
show_confusion_matrix(y_test, ensemble_preds, le, "Ensemble")
show_confusion_matrix(y_test_lr, lr_preds, le, "Logistic Regression")
show_confusion_matrix(y_test_knn, y_pred_knn, le_knn, "KNN")
show_confusion_matrix(y_true_poisson, y_pred_poisson, le_poisson, "Poisson")



