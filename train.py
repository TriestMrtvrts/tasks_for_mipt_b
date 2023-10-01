import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


!gdown 19ko8uq2u_5gtleTjByps1b3QeNtYG0s9
!unzip /content/mars-class.zip
!gdown 1J7pQpiXS-yJST2rbfRppEOjhl0combYC
!unzip /content/mars_final_private.zip
train_data = pd.read_csv('/content/mars-train-class.csv')
train_data.head(10)
test_data = pd.read_csv('/content/mars-private_test-class.csv')
test_data.head(10)
X_train = train_data.drop('Тип марсианина', axis=1)
y_train = train_data['Тип марсианина']
#X_test = test_data

ensemble_model = VotingClassifier(estimators=[
    ('xgb', XGBClassifier()),
    ('rf', RandomForestClassifier()),
    ('catboost', CatBoostClassifier()),
], voting='hard')

ensemble_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(ensemble_model, "./ens_mod_class.joblib")
