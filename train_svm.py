# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import joblib
# from backend.utils import load_data

# X_train, X_test, y_train, y_test = load_data()

# subset_size = min(10000, len(X_train))
# X_train_small = X_train.sample(subset_size, random_state=42)
# y_train_small = y_train.loc[X_train_small.index]

# model = SVC(kernel='rbf', random_state=42,probability=True)
# model.fit(X_train_small, y_train_small)
# y_pred = model.predict(X_test)

# print("SVM Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(4,3))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('SVM Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig("SVM_confusion_matrix.png")
# plt.close()

# joblib.dump(model, "backend/models/SVM_model.pkl")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from backend.utils import load_data

# Correct unpacking (only classification data)
X_train_c, X_test_c, y_train_c, y_test_c, _, _, _, _ = load_data()

# Optional sampling (if dataset is huge)
subset_size = min(10000, len(X_train_c))
X_train_small = X_train_c.sample(subset_size, random_state=42)
y_train_small = y_train_c.loc[X_train_small.index]

# SVM with probability=True for softmax output
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train_small, y_train_small)
y_pred = model.predict(X_test_c)

print("SVM Accuracy:", accuracy_score(y_test_c, y_pred))
print("Classification Report:\n", classification_report(y_test_c, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("SVM_confusion_matrix.png")
plt.close()

joblib.dump(model, "backend/models/SVM_model.pkl")

