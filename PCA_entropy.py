######################################################################################################
#  import libraries
######################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


######################################################################################################
#  def functions
######################################################################################################
def vec_density(d):
    d_for_hist = d * 10 ** precision
    hist = np.zeros(int(np.max(d_for_hist)) + 1)
    for val in d_for_hist:
        hist[int(val)] += 1
    return hist / np.sum(hist)


def entropy(fx):
    tmp = []
    for trial in np.arange(0, len(fx), 1):
        if fx[trial] == 0:
            tmp += [0]
        else:
            tmp += [-1 * fx[trial] * np.log2(fx[trial])]
    return np.sum(tmp)


def analyze_data(data):
    trials, features = np.shape(data)
    # calc variance
    var_data = np.var(data, axis=0)
    # calc entropy
    entropy_data_features = np.zeros(features)
    for feature in np.arange(0, features, 1):
        fx = vec_density(data[:, feature])
        entropy_data_features[feature] = entropy(fx)
    return var_data, entropy_data_features


######################################################################################################
#  gaussian data
######################################################################################################
# create data
trials = 1000
features = 10
precision = 2
m = 1000
mean = np.array([m, m, m, m, m, m, m, m, m, m])
c = 0  # no correlation between any two features!

for (v1, v2, v3, v4) in [(1, 1, 1, 1), (1, 10, 80, 200)]:

    cov = np.array([
        [v1, c, c, c, c, c, c, c, c, c],
        [c, v1, c, c, c, c, c, c, c, c],
        [c, c, v1, c, c, c, c, c, c, c],
        [c, c, c, v3, c, c, c, c, c, c],
        [c, c, c, c, v3, c, c, c, c, c],
        [c, c, c, c, c, v4, c, c, c, c],
        [c, c, c, c, c, c, v4, c, c, c],
        [c, c, c, c, c, c, c, v2, c, c],
        [c, c, c, c, c, c, c, c, v2, c],
        [c, c, c, c, c, c, c, c, c, v2],
    ])

    data = np.transpose(np.round(np.random.multivariate_normal(mean, cov, trials).T, decimals=precision))

    ### before PCA - var and entropy
    var_data, entropy_data_features = analyze_data(data)
    var_data_sum = np.sum(var_data)
    entropy_data_features_sum = np.sum(entropy_data_features)
    var_data_sorted_neg, entropy_data_features_sorted = (list(t) for t in
                                                         zip(*sorted(zip(-var_data, entropy_data_features))))
    var_data_sorted = -np.array(var_data_sorted_neg)

    ### after PCA - var and entropy
    pca = PCA(n_components=features)
    principalComponents = pca.fit(data)
    data_projected = pca.transform(data)
    var_data_projected, entropy_data_projected_features = analyze_data(data_projected)
    var_data_projected_sum = np.sum(var_data_projected)
    entropy_data_projected_features_sum = np.sum(entropy_data_projected_features)

    print("entropy before PCA:", entropy_data_features_sum)
    print("variance before PCA:", var_data_sum)
    print("entropy after PCA:", entropy_data_projected_features_sum)
    print("variance after PCA:", var_data_projected_sum)

    ### plotting the variance and the entropy by features, variance-sorted
    features_vec = np.arange(0, 10, 1)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('feature')
    ax1.set_ylabel('variance')
    ax1.plot(features_vec, var_data_sorted, color="red", marker='o', label="variance-before")
    ax1.plot(features_vec, var_data_projected, color="red", marker='D', label="variance-after")
    ax1.tick_params(axis='y', labelcolor="red")
    ax1.legend(loc=0)

    ax2 = ax1.twinx()  # shared x axis.
    ax2.set_ylabel('entropy[bits]')
    ax2.plot(features_vec, entropy_data_features_sorted, color="blue", marker='o', label="entropy-before")
    ax2.plot(features_vec, entropy_data_projected_features, color="blue", marker='D', label="entropy-after")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.legend(loc=3)

    plt.title(
        "Variance and Entropy for variance-sorted features\n before and after PCA, for (v1,v2,v3,v3)=(" + str(v1)+ "," + str(
            v2) + "," + str(v3) + "," + str(v4) + ")")
    plt.show()

    ### plotting entropy as a function of variance
    plt.scatter(var_data_sorted, entropy_data_features_sorted, marker="o", label="before")
    plt.scatter(var_data_projected, entropy_data_projected_features, marker="o", label="after")
    plt.title("Entropy as a function of variance before and after PCA,\nfor (v1,v2,v3,v3)=(" + str(v1)+ "," + str(
            v2) + "," + str(v3) + "," + str(v4) + ")")
    plt.xlabel("variance")
    plt.ylabel("entropy[bits]")
    plt.legend()
    plt.show()

    ### plotting the accomulating variance and entropy over features, variance-sorted
    entropy_before_sum = []
    entropy_after_sum = []
    variance_before_sum = []
    variance_after_sum = []
    for i in np.arange(1, features + 1, 1):
        entropy_before_sum += [np.sum(entropy_data_features_sorted[0:i])]
        entropy_after_sum += [np.sum(entropy_data_projected_features[0:i])]
        variance_before_sum += [np.sum(var_data_sorted[0:i])]
        variance_after_sum += [np.sum(var_data_projected[0:i])]

    features_vec = np.arange(0, 10, 1)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('feature')
    ax1.set_ylabel('variance')
    ax1.plot(features_vec, variance_before_sum, color="red", marker='o', label="variance-before")
    ax1.plot(features_vec, variance_after_sum, color="red", marker='D', label="variance-after")
    ax1.tick_params(axis='y', labelcolor="red")
    ax1.legend(loc=2)

    ax2 = ax1.twinx()  # shared x axis.
    ax2.set_ylabel('entropy')
    ax2.plot(features_vec, entropy_before_sum, color="blue", marker='o', label="entropy-before")
    ax2.plot(features_vec, entropy_after_sum, color="blue", marker='D', label="entropy-after")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.legend(loc=4)

    plt.title(
        "Accomulating variance-sum and entropy-sum for variance-sorted\nfeatures before and after PCA, for (v1,v2,v3,v3)=(" + str(v1)+ "," + str(
            v2) + "," + str(v3) + "," + str(v4) + ")")
    plt.show()
