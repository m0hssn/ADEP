import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def extract_pair_latent_representation(model, drug_pair, df_drug):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    pair1 = drug_pair[0]
    pair2 = drug_pair[1]

    drugA = torch.tensor(df_drug[df_drug['name'] == pair1['drugA']].drop(columns=['name']).values.astype('float32'))
    drugB = torch.tensor(df_drug[df_drug['name'] == pair1['drugB']].drop(columns=['name']).values.astype('float32'))
    pairA = torch.cat([(drugA), (drugB)]).flatten()

    drugC = torch.tensor(df_drug[df_drug['name'] == pair2['drugA']].drop(columns=['name']).values.astype('float32'))
    drugD = torch.tensor(df_drug[df_drug['name'] == pair2['drugB']].drop(columns=['name']).values.astype('float32'))
    pairB = torch.cat([(drugC), (drugD)]).flatten()

    pairA = pairA.to(device)
    pairB = pairB.to(device)

    with torch.no_grad():
        latentA = model.encoder(pairA.view(1, -1))
        latentB = model.encoder(pairB.view(1, -1))

    return latentA, latentB


def create_similarity_matrix(model, drug_pairs):
    print('in csm ...')
    num_rows = int(len(drug_pairs) * 0.001)

    drug_pairs = drug_pairs.head(num_rows)
    num_pairs = len(drug_pairs)
    similarity_matrix = np.zeros((num_pairs, num_pairs))
    d = num_pairs * [0]
    s = []
    for i in range(num_pairs):
        s.append(d)
    similarity_sides = s
    for i, rowa in drug_pairs.iterrows():
        for j, rowb in drug_pairs.iterrows():
            latentA, latentB = extract_pair_latent_representation(model, [rowa, rowb])
            similarity_matrix[i, j] = cosine_similarity(latentA.to('cpu'), latentB.to('cpu'))
            similarity_sides[i][j] = [rowa['side'], rowb['side'], latentA.to('cpu'), latentB.to('cpu')]
    return similarity_matrix, np.array(similarity_sides)

def extract_similar_pairs(similarity_matrix, similarity_sides, threshold):
    print('in esp ...')
    similar_pairs_laten = []
    sides_compare = []
    compare_results = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] >= threshold:
                sidea, sideb, latenta, latentb = similarity_sides[i][j]
                if latenta.tolist() not in similar_pairs_laten:
                    similar_pairs_laten.append(latenta.tolist())
                if latentb.tolist() not in similar_pairs_laten:
                    similar_pairs_laten.append(latentb.tolist())
                compare_results.append(f'pair{i} and pair{j} with similarity {similarity_matrix[i, j]} have sides in this order {sidea} , {sideb}')
    return np.array(similar_pairs_laten), compare_results


def visualize_latent(latent_representations, method='pca'):
    print('in vl ...')
    reducer = None
    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=2)

    latent_representations = latent_representations.reshape(latent_representations.shape[0], -1)
    reduced_latent = reducer.fit_transform(latent_representations)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(reduced_latent[:, 0], reduced_latent[:, 1], reduced_latent[:, 2])
    ax.set_title(f'Latent Representations Visualization ({method.capitalize()})')

    plt.show()
