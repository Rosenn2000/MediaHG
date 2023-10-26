import pandas as pd
s=pd.read_csv("process.csv")
v=pd.DataFrame(s)
temp_df=v.sample(n=6000)
temp_df.to_csv("6000.csv")
