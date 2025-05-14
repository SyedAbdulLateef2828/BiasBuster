import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BiasBuster:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def summary(self):
        print("\n🔍 Dataset Summary:")
        print(self.df.describe(include='all'))

    def class_distribution(self, column):
        print(f"\n📊 Distribution in '{column}':")
        print(self.df[column].value_counts())

        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x=column)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        plt.savefig(f"{column}_distribution.png")
        plt.close()

    def correlation_heatmap(self):
        print("\n📈 Correlation Matrix:")
        numeric_df = self.df.select_dtypes(include='number')
        corr = numeric_df.corr()
        print(corr)

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

    def detect_imbalance(self, target_column):
        print(f"\n⚖️ Checking imbalance in target column: '{target_column}'")
        counts = self.df[target_column].value_counts(normalize=True) * 100
        for cls, pct in counts.items():
            print(f" - {cls}: {pct:.2f}%")
        if counts.max() > 75:
            print("⚠️ Potential class imbalance detected.")

