from biasbuster import BiasBuster

analyzer = BiasBuster("sample_data.csv")
analyzer.summary()
analyzer.class_distribution("gender")
analyzer.class_distribution("approved")
analyzer.correlation_heatmap()
analyzer.detect_imbalance("approved")
