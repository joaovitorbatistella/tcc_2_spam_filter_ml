import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Basic Logistic/Sigmoid Function
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))  # Sigmoid function

ax1.plot(x, y, 'b-', linewidth=2, label='Sigmoid Function')
ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('x (Linear Combination)', fontsize=12)
ax1.set_ylabel('P(y=1|x)', fontsize=12)
ax1.set_title('Logistic/Sigmoid Function', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(-0.05, 1.05)

# Add annotations
ax1.annotate('P → 1', xy=(5, 0.99), xytext=(6, 0.85),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
            fontsize=10, color='blue')
ax1.annotate('P → 0', xy=(-5, 0.01), xytext=(-6, 0.15),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
            fontsize=10, color='blue')

# Plot 2: Logistic Regression with Sample Data
# Generate sample data
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=1, n_redundant=0, 
                          n_informative=1, n_clusters_per_class=1, 
                          random_state=42)

# Fit logistic regression
model = LogisticRegression()
model.fit(X, y)

# Create smooth curve for plotting
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(X_plot)[:, 1]

# Plot the data points
colors = ['red', 'blue']
for i in range(2):
    mask = y == i
    ax2.scatter(X[mask], y[mask], c=colors[i], alpha=0.6, 
               label=f'Class {i}', s=50)

# Plot the logistic regression curve
ax2.plot(X_plot, y_prob, 'g-', linewidth=2, label='Logistic Regression')
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
           label='Decision Boundary')

ax2.set_xlabel('Feature Value', fontsize=12)
ax2.set_ylabel('Probability / Class', fontsize=12)
ax2.set_title('Logistic Regression on Sample Data', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(-0.1, 1.1)

# Add equation text
equation_text = r'$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$'
fig.suptitle(f'Logistic Regression: {equation_text}', fontsize=16, y=0.95)

plt.tight_layout()

dirname = os.path.dirname(__file__)
_datetime               = datetime.now().strftime("%Y%m%d-%H%M")
base_path               = f"{dirname}/../output/vizualizations/{_datetime}"
os.makedirs(base_path, exist_ok=True)

# Exibir o heatmap
plt.savefig(f"{base_path}/scatter.png")

# Print model coefficients
print("Logistic Regression Model:")
print(f"Intercept (β₀): {model.intercept_[0]:.3f}")
print(f"Coefficient (β₁): {model.coef_[0][0]:.3f}")
print(f"Equation: P(y=1|x) = 1 / (1 + exp(-({model.intercept_[0]:.3f} + {model.coef_[0][0]:.3f}*x)))")

# Demonstrate predictions
print("\nSample Predictions:")
test_points = np.array([[-2], [0], [2]]).reshape(-1, 1)
probabilities = model.predict_proba(test_points)
predictions = model.predict(test_points)

for i, (point, prob, pred) in enumerate(zip(test_points.flatten(), probabilities, predictions)):
    print(f"x = {point:4.1f}: P(y=1) = {prob[1]:.3f}, Predicted class = {pred}")