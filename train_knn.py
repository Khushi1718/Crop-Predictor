# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import joblib
# from backend.utils import load_data

# X_train, X_test, y_train, y_test = load_data()

# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# print("KNN Accuracy:", acc)
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(4,3))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
# plt.title('KNN Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig("KNN_confusion_matrix.png")
# plt.close()

# # model
# joblib.dump(model, "backend/models/KNN_model.pkl")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from backend.utils import load_data

# load classification data only
X_train_c, X_test_c, y_train_c, y_test_c, _, _, _, _ = load_data()

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_c, y_train_c)
y_pred = model.predict(X_test_c)

acc = accuracy_score(y_test_c, y_pred)
print("KNN Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test_c, y_pred))

# confusion matrix
cm = confusion_matrix(y_test_c, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("KNN_confusion_matrix.png")
plt.close()

# save model
joblib.dump(model, "backend/models/KNN_model.pkl")
print("Model saved as backend/models/KNN_model.pkl")
