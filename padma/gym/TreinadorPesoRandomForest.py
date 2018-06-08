from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils import df_plot, monta_df, reg_plot
import matplotlib.pyplot as plt 
import numpy as np

BINS = 16
DIR = r'D:\Users\25052288840\Downloads\imgs'
MAX_ROWS = 500

df, imagens = monta_df(BINS, from_dir=DIR, max_rows=MAX_ROWS)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[:15]], df['peso'], test_size=0.25, random_state=42)
print(df.columns)


forest = RandomForestRegressor(max_depth=10, random_state=0)
forest.fit(X_train, y_train)

reg_plot(forest, X_test, y_test)
print(len(X_test))



fig=plt.figure(figsize=(16, 40))
columns = 4
rows = 10

indexes = np.random.randint(len(imagens) - 1, size=columns*rows + 1)
print(indexes)
random_histograms = [np.histogram(np.asarray(imagens[index][0]), bins=BINS)[0][:15] for index in indexes]
random_predicts = forest.predict(random_histograms)
for i in range(1, columns*rows +1):
    img = imagens[indexes[i]][0]
    ax = fig.add_subplot(rows, columns, i)
    title = ' %d X %d ' % (df['peso'].iloc[indexes[i]],
                            random_predicts[i])
    ax.set(title=title )
    plt.imshow(img)
plt.show()