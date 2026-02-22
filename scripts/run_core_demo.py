from microtest.core import add, multiply
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from microtest.core import entropy
df = pd.read_csv(r"C:\\Users\\amoh\\Desktop\\microtox\\data\\processed\\treadmill\\P001_L0_T1.csv")


gyro_x = df["IMU-1 Gyr.X"]
entropy_value = entropy(gyro_x)

print("Shannon Entropy:", entropy_value)

# if __name__ == "__main__":
#     main()