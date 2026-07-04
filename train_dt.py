# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import joblib
# from backend.utils import load_data

# X_train, X_test, y_train, y_test = load_data()

# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(4,3))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
# plt.title('Decision Tree Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig("DecisionTree_confusion_matrix.png")
# plt.close()

# # model
# joblib.dump(model, "backend/models/DecisionTree_model.pkl")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from backend.utils import load_data

X_train_c, X_test_c, y_train_c, y_test_c, _, _, _, _ = load_data()

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_c, y_train_c)
y_pred = model.predict(X_test_c)

print("Decision Tree Accuracy:", accuracy_score(y_test_c, y_pred))
print("Classification Report:\n", classification_report(y_test_c, y_pred))

cm = confusion_matrix(y_test_c, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("DecisionTree_confusion_matrix.png")
plt.close()

joblib.dump(model, "backend/models/DecisionTree_model.pkl")
print("Saved: backend/models/DecisionTree_model.pkl")
