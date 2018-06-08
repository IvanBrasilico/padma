from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils import df_plot, monta_df, reg_plot

BINS = 16
DIR = r'D:\Users\25052288840\Downloads\imgs'

df, imagens = monta_df(BINS, from_dir=DIR)
df.head()

"""
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[:15]], df['peso'], test_size=0.25, random_state=42)
print(df.columns)


forest = RandomForestRegressor(max_depth=10, random_state=0)
forest.fit(X_train, y_train)

plt = reg_plot(forest, X_test, y_test)
print(len(X_test))
"""