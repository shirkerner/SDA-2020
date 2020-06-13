######################################################################################################
#  import libraries
######################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


######################################################################################################
#  def functions
######################################################################################################
def vec_density(d, max):
    hist = np.zeros(max + 1)
    for val in d:
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
    var_data_sum = np.sum(var_data)
    # calc entropy
    entropy_data_features = np.zeros(features)
    for feature in np.arange(0, features, 1):
        max = np.int(np.max(data))
        fx = vec_density(data[:, feature], max)
        entropy_data_features[feature] = entropy(fx)
    entropy_data_features_sum = np.sum(entropy_data_features)
    return var_data_sum, entropy_data_features_sum


######################################################################################################
#  gaussian data
######################################################################################################
# create data
trials = 1000
features = 10
data = np.ones([trials, features])
for i in np.arange(0, trials, 1):
    for j in np.arange(0, features, 1):
        data[i][j] = int(np.round(np.random.normal(1000, 10)))

show_graphs = False
### before PCA - var and entropy
var_data_sum, entropy_data_features_sum = analyze_data(data)

### after PCA - var and entropy
pca = PCA(n_components=features)
principalComponents = pca.fit(data)
data_projected = pca.transform(data)
var_data_projected_sum, entropy_data_projected_features_sum = analyze_data(data_projected)

print("Gaussian data:")
print("entropy before PCA:", entropy_data_features_sum)
print("variance before PCA:", var_data_sum)
print("entropy after PCA:", entropy_data_projected_features_sum)
print("variance after PCA:", var_data_projected_sum)

######################################################################################################
#  Bernoulli data
######################################################################################################
# p = 0.5
trials = 1000
features = 10
entropy_ratio_vec = []
variance_subtraction_vec = []
p_vec = np.arange(0.01, 0.99, 0.05)

for p in p_vec:
    # create data
    data = np.ones([trials, features])
    for i in np.arange(0, trials, 1):
        for j in np.arange(0, features, 1):
            if np.random.rand() < p:
                data[i][j] = 1000
            else:
                data[i][j] = 10

    show_graphs = True
    ### before PCA - var and entropy
    var_data_sum, entropy_data_features_sum = analyze_data(data)

    ### after PCA - var and entropy
    pca = PCA(n_components=features)
    principalComponents = pca.fit(data)
    data_projected = pca.transform(data)
    var_data_projected_sum, entropy_data_projected_features_sum = analyze_data(data_projected)

    entropy_ratio = entropy_data_features_sum / entropy_data_projected_features_sum
    entropy_ratio_vec += [entropy_ratio]
    variance_subtraction = var_data_sum - var_data_projected_sum
    variance_subtraction_vec += [variance_subtraction]

plt.plot(p_vec, entropy_ratio_vec)
plt.title("Bernoulli data: Entropy-total-sum ratio before/after PCA\nover different p values")
plt.ylim(0, 0.2)
plt.ylabel("entropy ratio before/after")
plt.xlabel("p")
plt.show()

plt.plot(p_vec, variance_subtraction_vec)
plt.title("Bernoulli data: Variance-total-sum substraction before minus after PCA\nover different p values")
plt.ylim(-0.1, 0.1)
plt.ylabel("variance Subtraction before minus after [AU]")
plt.xlabel("p")
plt.show()
