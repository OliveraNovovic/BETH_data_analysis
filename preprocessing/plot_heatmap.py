import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(df, data_name):
    data = df.drop(['evil'], axis=1)
    plt.subplots(figsize=(15, 10))
    corr = data.corr() #Pearson correlation
    ax = sns.heatmap(corr, cmap="Blues", annot=True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
        )
    ax.set_title("Pearson correlation heatmap - " + data_name, fontsize=20)
    plt.subplots_adjust(bottom=0.30)
    plt.show()