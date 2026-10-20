
# Assignment 5: Regression Techniques
# Linear Regression and Polynomial Regression

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 91


# Step 2: Load and preprocess the dataset
df = pd.read_csv("housing.xls")

print(df.head())
print("Number of samples:", df.shape[0])
print("Number of features:", df.shape[1])

# Remove longitude and latitude, as in the original notebook
df.drop(columns=["longitude", "latitude"], inplace=True)

# Fill missing numerical values
df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)

# Encode categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

# Target feature
output_feature = "median_house_value"


# Step 3: Correlation analysis
correlation_matrix = df.corr(numeric_only=True)[
    [output_feature]
].sort_values(by=output_feature, ascending=False)

top_feature = correlation_matrix.index[1]
secnd_top_feature = correlation_matrix.index[2]

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Step 4: Linear Regression
X = df[[top_feature]]
y = df[output_feature]

# Fixed 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

print("\nLINEAR REGRESSION")
print("Feature:", top_feature)
print("Coefficient:", np.round(linear_model.coef_, 2))
print("Intercept:", round(float(linear_model.intercept_), 2))

# Linear Regression metrics
Lin_train_mse = mean_squared_error(y_train, y_train_pred)
Lin_train_mae = mean_absolute_error(y_train, y_train_pred)
Lin_train_r2 = r2_score(y_train, y_train_pred)

Lin_test_mse = mean_squared_error(y_test, y_test_pred)
Lin_test_mae = mean_absolute_error(y_test, y_test_pred)
Lin_test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining Performance")
print("MSE:", round(Lin_train_mse, 3))
print("MAE:", round(Lin_train_mae, 3))
print("R2 Score:", round(Lin_train_r2, 3))

print("\nTesting Performance")
print("MSE:", round(Lin_test_mse, 3))
print("MAE:", round(Lin_test_mae, 3))
print("R2 Score:", round(Lin_test_r2, 3))


# Step 5: Linear Regression visualization
plt.figure(figsize=(10, 6))

plt.scatter(X_train[top_feature], y_train,
            color="green", s=10, alpha=0.5, label="Training Data")

plt.scatter(X_test[top_feature], y_test,
            color="blue", s=10, alpha=0.5, label="Testing Data")

# Sort feature values so the regression line is displayed correctly
sorted_indices = np.argsort(X_test[top_feature].values)

plt.plot(
    X_test[top_feature].values[sorted_indices],
    y_test_pred[sorted_indices],
    color="red",
    linewidth=2,
    label="Linear Regression Line"
)

plt.xlabel(top_feature)
plt.ylabel(output_feature)
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()


# Step 6: Polynomial Regression
X_poly_data = df[[top_feature, secnd_top_feature]]
y_poly_data = df[output_feature]

# Fixed 80:20 train-test split
X_train_poly_data, X_test_poly_data, y_train_poly, y_test_poly = (
    train_test_split(
        X_poly_data, y_poly_data,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
)

# Generate polynomial features of degree 2
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_poly_data)
X_test_poly = poly.transform(X_test_poly_data)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)

y_train_pred_poly = poly_model.predict(X_train_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)

# Polynomial Regression metrics
poly_train_mse = mean_squared_error(y_train_poly, y_train_pred_poly)
poly_train_mae = mean_absolute_error(y_train_poly, y_train_pred_poly)
poly_train_r2 = r2_score(y_train_poly, y_train_pred_poly)

poly_test_mse = mean_squared_error(y_test_poly, y_test_pred_poly)
poly_test_mae = mean_absolute_error(y_test_poly, y_test_pred_poly)
poly_test_r2 = r2_score(y_test_poly, y_test_pred_poly)

print("\nPOLYNOMIAL REGRESSION")

print("\nTraining Performance")
print("MSE:", round(poly_train_mse, 3))
print("MAE:", round(poly_train_mae, 3))
print("R2 Score:", round(poly_train_r2, 3))

print("\nTesting Performance")
print("MSE:", round(poly_test_mse, 3))
print("MAE:", round(poly_test_mae, 3))
print("R2 Score:", round(poly_test_r2, 3))


# Step 7: Polynomial Regression 3D visualization
x1_min, x1_max = (
    X_poly_data[top_feature].min(),
    X_poly_data[top_feature].max()
)

x2_min, x2_max = (
    X_poly_data[secnd_top_feature].min(),
    X_poly_data[secnd_top_feature].max()
)

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 50),
    np.linspace(x2_min, x2_max, 50)
)

grid_points = np.c_[xx1.ravel(), xx2.ravel()]
grid_points_poly = poly.transform(grid_points)

yy_pred = poly_model.predict(grid_points_poly)
yy_pred = yy_pred.reshape(xx1.shape)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X_test_poly_data[top_feature],
    X_test_poly_data[secnd_top_feature],
    y_test_poly,
    color="red",
    s=5,
    label="Actual Test Data"
)

ax.plot_surface(
    xx1, xx2, yy_pred,
    cmap="viridis",
    alpha=0.5
)

ax.set_xlabel(top_feature)
ax.set_ylabel(secnd_top_feature)
ax.set_zlabel(output_feature)
ax.set_title("Polynomial Regression Surface")
ax.legend()
plt.tight_layout()
plt.show()


# Step 8: Predict house value from user input
user_input_1 = float(input(f"Enter {top_feature}: "))
user_input_2 = float(input(f"Enter {secnd_top_feature}: "))

user_data = pd.DataFrame({
    top_feature: [user_input_1],
    secnd_top_feature: [user_input_2]
})

user_data_poly = poly.transform(user_data)
prediction = poly_model.predict(user_data_poly)

print("\nPredicted House Value:", round(prediction[0], 2))


# Step 9: Compare both models
comparison = pd.DataFrame({
    "Metric": ["MSE", "MAE", "R2 Score"],

    "Linear Regression - Training": [
        Lin_train_mse, Lin_train_mae, Lin_train_r2
    ],

    "Linear Regression - Testing": [
        Lin_test_mse, Lin_test_mae, Lin_test_r2
    ],

    "Polynomial Regression - Training": [
        poly_train_mse, poly_train_mae, poly_train_r2
    ],

    "Polynomial Regression - Testing": [
        poly_test_mse, poly_test_mae, poly_test_r2
    ]
})

print("\nMODEL COMPARISON")
print(comparison.round(3).to_string(index=False))


# Step 10: Compare testing performance visually
metrics = ["MSE", "MAE", "R2 Score"]

linear_values = [Lin_test_mse, Lin_test_mae, Lin_test_r2]
poly_values = [poly_test_mse, poly_test_mae, poly_test_r2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    bars = ax.bar(
        ["Linear", "Polynomial"],
        [linear_values[i], poly_values[i]],
        color=["skyblue", "salmon"],
        width=0.5
    )

    ax.set_title("Comparison of " + metrics[i])
    ax.set_ylabel(metrics[i])
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

plt.suptitle("Linear vs Polynomial Regression - Testing Performance")
plt.tight_layout()
plt.show()
