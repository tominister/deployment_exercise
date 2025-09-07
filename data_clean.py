import pandas as pd

df= pd.read_csv("Electronic_sales.csv")
df=df.sample(n=1000, random_state=42)
df['target'] = (df['Order Status']== 'Completed').astype(int)
df= df.drop(columns=['Customer ID', 'SKU', 'Purchase Date', 'Order Status'])




df = df.drop(columns=['Add-ons Purchased', 'Add-on Total'])
df = pd.get_dummies(df, drop_first=True)
df = df.replace({True: 1, False: 0})

# Remove duplicate rows
df = df.drop_duplicates()
df.to_csv("Electronic_sales_cleaned.csv", index=False)