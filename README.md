# GEAT
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# Load the dataset
df = pd.read_csv("/content/Breast_GSE45827.csv")

# Perform mean imputation
df_imputed = df.copy()
for column in df_imputed.select_dtypes(include=[np.number]).columns:
    mean_value = df_imputed[column].mean()
    df_imputed[column].fillna(mean_value, inplace=True)

# Drop the 'samples' column and separate features and target
data = df_imputed.drop(columns=['samples'])
X = data.drop(columns=['type'])
y = data['type']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply PCA
n_components = min(len(X), len(X.columns), 20)
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=42)

# Fit a RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')

# Compute and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Save the trained model to a file
with open('gene_expression_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Load the CSV file into a DataFrame
data = pd.read_csv("/content/Breast_GSE45827.csv")

# Impute missing values
for column in data.select_dtypes(include=[np.number]).columns:
    mean_value = data[column].mean()
    data[column].fillna(mean_value, inplace=True)

# Standardize the data
scaled_data = scaler.transform(data.drop(columns=['samples', 'type']))

# Reduce dimensionality using PCA
reduced_data = pca.transform(scaled_data)

# Make predictions
predictions = classifier.predict(reduced_data)

# Decode the predictions
decoded_predictions = label_encoder.inverse_transform(predictions)

# Add predictions to the DataFrame
data['predicted_type'] = decoded_predictions

# Print the predictions
print("Predictions:")
print(data[['samples', 'type', 'predicted_type']])

# Check the shape of the DataFrame
print(df.shape)

APP GUI
Here is the formatted and organized code without comments:

```python
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import threading

class GeneExpressionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Gene Expression Analysis Tool")
        self.master.geometry("500x400")
        self.font_style = ("Calibri", 12)

        self.file_path = None

        self.load_data_button = tk.Button(master, text="Load Data", command=self.load_data, bg="#4CAF50", fg="white", font=self.font_style)
        self.load_data_button.pack(pady=10)

        self.progress_style = ttk.Style()
        self.progress_style.theme_use('default')
        self.progress_style.configure("Color.Horizontal.TProgressbar", foreground='#FFD700', background='#FFD700')
        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=200, mode="indeterminate", style="Color.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)

        self.run_analysis_button = tk.Button(master, text="Run Analysis", command=self.run_analysis, bg="#008CBA", fg="white", font=self.font_style)
        self.run_analysis_button.pack(pady=10)

        self.results_text = tk.Text(master, height=10, width=50, wrap="word", font=self.font_style)
        self.results_text.pack(pady=10)
        self.results_text.insert(tk.END, "Results will be displayed here.")

    def load_data(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            messagebox.showinfo("Success", "Data loaded successfully from: {}".format(self.file_path))
        else:
            messagebox.showerror("Error", "No file selected.")

    def run_analysis(self):
        if self.file_path:
            self.progress_bar.start()
            threading.Thread(target=self.analyze_data).start()
        else:
            messagebox.showerror("Error", "Please load data first.")

    def analyze_data(self):
        try:
            df = pd.read_csv(self.file_path)
            df_imputed = self.impute_missing_values(df)
            X_reduced, pca, classifier = self.load_model_and_transform_data(df_imputed)
            predictions = self.make_predictions(X_reduced, classifier)
            df['predicted_type'] = self.label_encoder.inverse_transform(predictions)
            self.display_results(df[['samples', 'type', 'predicted_type']])
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.progress_bar.stop()

    def impute_missing_values(self, df):
        df_imputed = df.copy()
        for column in df_imputed.select_dtypes(include=[np.number]).columns:
            mean_value = df_imputed[column].mean()
            df_imputed[column].fillna(mean_value, inplace=True)
        return df_imputed

    def load_model_and_transform_data(self, df):
        with open('gene_expression_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.drop(columns=['samples', 'type']))
        pca = PCA(n_components=min(len(df), len(df.columns), 20))
        X_reduced = pca.fit_transform(X_scaled)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
        return X_reduced, pca, classifier

    def make_predictions(self, X_reduced, classifier):
        predictions = classifier.predict(X_reduced)
        return predictions

    def display_results(self, results_df):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_df.to_string(index=False))

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    style.configure("TButton", font=("Calibri", 12))
    app = GeneExpressionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
```
