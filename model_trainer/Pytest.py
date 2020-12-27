import re
import numpy as np
import pandas as pd
import config
data = pd.read_csv(config.SUBMITCSV)
print(data.head())
