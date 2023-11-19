import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import joblib


dataset=pd.read_csv('loan_approval_dataset (1).csv')

#removing loan_id column
dataset = dataset.drop('loan_id', axis=1)

#replacing missing values with mean for numerical data
numerical_cols = ['no_of_dependents','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']
for col in numerical_cols:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

#replacing missing values with mode for categorical data
categorical_cols = ['education', 'self_employed']
dataset[categorical_cols] = dataset[categorical_cols].fillna(dataset[categorical_cols].mode().iloc[0])

#removing duplicate rows
dataset = dataset[~dataset.duplicated()]
dataset.shape

dataset['education'] = dataset['education'].map({' Graduate': 1, ' Not Graduate': 0})
dataset['self_employed'] = dataset['self_employed'].map({' Yes': 1, ' No': 0})

#splitting input and output features
X = dataset.drop('loan_status', axis=1)
y = dataset['loan_status']

#splitting training and testing (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaling input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.joblib')

#CNN + HYBRID

# Encode target labels using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

joblib.dump(label_encoder, 'label_encoder.joblib')

# Convert X_train to a numpy array
X_train_array = X_train_scaled

# Split the data into train and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_array, y_train_encoded, test_size=0.1, random_state=42)

# Pad the input sequences to a desired length (e.g., 30)
max_sequence_length = 30
X_train_split = pad_sequences(X_train_split, maxlen=max_sequence_length, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=max_sequence_length, padding='post', truncating='post')

# Convert target labels to one-hot encoded format
num_classes = len(label_encoder.classes_)
y_train_split_encoded = to_categorical(y_train_split, num_classes)
y_val_encoded = to_categorical(y_val, num_classes)

# Assuming X_test is a DataFrame, convert it to a numpy array and pad it
X_test_array = X_test.to_numpy()
X_test_padded = pad_sequences(X_test_array, maxlen=max_sequence_length, padding='post', truncating='post')

# Define the CNN model
cnn_model = Sequential([
    # Convolutional layer with 128 filters and a kernel size of 3, using ReLU activation function
    Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(max_sequence_length, 1)),
    # Convolutional layer with 128 filters and a kernel size of 3, using ReLU activation function
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    # Max pooling layer with a pool size of 2
    MaxPooling1D(pool_size=2),
    # Dropout layer with a dropout rate of 0.25
    Dropout(0.25),
    # Convolutional layer with 256 filters and a kernel size of 3, using ReLU activation function
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    # Convolutional layer with 256 filters and a kernel size of 3, using ReLU activation function
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    # Max pooling layer with a pool size of 2
    MaxPooling1D(pool_size=2),
    # Dropout layer with a dropout rate of 0.25
    Dropout(0.25),
    # Flatten layer to transform the 3D output to 1D
    Flatten(),
    # Fully connected Dense layer with 512 units and ReLU activation function
    Dense(512, activation='relu'),
    # Dropout layer with a dropout rate of 0.5
    Dropout(0.5),
    # Fully connected Dense layer with 256 units and ReLU activation function
    Dense(256, activation='relu'),
    # Dropout layer with a dropout rate of 0.5
    Dropout(0.5),
    # Output layer with 'num_classes' units and softmax activation function for multi-class classification
    Dense(num_classes, activation='softmax')
])

# Generate predictions using the trained CNN model on the training and validation sets
cnn_train_output = cnn_model.predict(X_train_split)
cnn_val_output = cnn_model.predict(X_val)

# Ensure that the CNN model's output is properly configured for binary classification
# Assuming the CNN output has two units (one for each class)
# If not, adjust the CNN model's output layer accordingly

# Flatten the CNN model output to a 1D array, assuming the positive class is the second unit
cnn_train_output_flat = cnn_train_output[:, 1]
cnn_val_output_flat = cnn_val_output[:, 1]

cnn_model.save('cnn_model.h5')

# Define the LightGBM model parameters for binary classification
lgb_params = {
    'objective': 'binary',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.005,
    'n_estimators': 1000,
}

# Create the LightGBM model
lgb_model = lgb.LGBMClassifier(**lgb_params)

# Create DataFrames from numpy arrays for concatenation with specified column names
X_train_split_df = pd.DataFrame(X_train_split, columns=["feature_" + str(i) for i in range(X_train_split.shape[1])])
X_val_df = pd.DataFrame(X_val, columns=["feature_" + str(i) for i in range(X_val.shape[1])])

# Combine the original features with the CNN output for training and validation sets
X_train_combined = pd.concat([X_train_split_df, pd.Series(cnn_train_output_flat, name='cnn_output')], axis=1)
X_val_combined = pd.concat([X_val_df, pd.Series(cnn_val_output_flat, name='cnn_output')], axis=1)

joblib.dump(lgb_model, 'lgb_model.joblib')


# Train the LightGBM model on the combined data
lgb_model.fit(X_train_combined, y_train_split)

# Evaluate the hybrid model on the validation set
# Compute accuracy and log loss
val_accuracy = lgb_model.score(X_val_combined, y_val)
val_loss = log_loss(y_val, lgb_model.predict_proba(X_val_combined))

# Print the evaluation metrics
print("Validation Accuracy:", val_accuracy)
print("Validation Loss:", val_loss)

val_pred = lgb_model.predict(X_val_combined)
# Calculate the accuracy of the hybrid model
hybrid_model_accuracy = accuracy_score(y_val, val_pred)
print("Hybrid Model Accuracy:", hybrid_model_accuracy)

joblib.dump(lgb_model, 'hybrid_model.joblib')