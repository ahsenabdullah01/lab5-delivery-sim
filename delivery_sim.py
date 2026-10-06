import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

normal_times = np.random.normal(
    loc=32,
    scale=7,
    size=97
)

normal_times = np.round(normal_times, 1)

delivery_times = np.concatenate([
    normal_times,
    [68, 71, 75]
])

delivery_times = np.round(delivery_times, 1)

data = {
    "OrderID": range(1, 101),
    "DeliveryTime(min)": delivery_times
}

pd.DataFrame(data).to_csv(
    "delivery_times.csv",
    index=False
)

df = pd.read_csv("delivery_times.csv")

delivery = df["DeliveryTime(min)"]

mean_time = delivery.mean()
std_time = delivery.std()

print(
    f"Real delivery times -- Mean: "
    f"{mean_time:.2f} min Std: {std_time:.2f} min"
)

simulated_times = np.random.normal(
    loc=mean_time,
    scale=std_time,
    size=100
)

print(
    f"Simulated 100 values from "
    f"Normal(mean={mean_time:.2f}, std={std_time:.2f})"
)

print("\nMissing values:", delivery.isnull().sum())

q1 = delivery.quantile(0.25)
median = delivery.quantile(0.50)
q3 = delivery.quantile(0.75)

print(
    f"\nPercentiles -- 25th: {q1:.2f} "
    f"50th: {median:.2f} "
    f"75th: {q3:.2f}"
)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(
    f"IQR outlier bounds: "
    f"{lower_bound:.2f} - {upper_bound:.2f}"
)

outliers = df[
    (df["DeliveryTime(min)"] < lower_bound) |
    (df["DeliveryTime(min)"] > upper_bound)
]

print(
    "\nOutliers detected:",
    len(outliers)
)

print(
    "Outlier delivery times:",
    outliers["DeliveryTime(min)"].tolist()
)

percentage_outliers = (
    len(outliers) / len(df)
) * 100

print(
    f"Percentage flagged as outliers: "
    f"{percentage_outliers:.2f}%"
)

plt.figure(figsize=(9, 5))

plt.hist(
    delivery,
    bins=15,
    alpha=0.6,
    label="Real Delivery Times"
)

plt.hist(
    simulated_times,
    bins=15,
    alpha=0.6,
    label="Simulated Normal Times"
)

plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()

plt.savefig(
    "delivery_comparison_histogram.png"
)

plt.show()

plt.figure(figsize=(6, 5))

plt.boxplot(delivery)

plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")

plt.savefig("delivery_boxplot.png")

plt.show()

print(
    "\nSaved delivery_comparison_histogram.png"
)

print("Saved delivery_boxplot.png")