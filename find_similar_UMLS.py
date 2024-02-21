import pandas as pd
import numpy as np
import faiss

# Funktion, um die Embeddings und CUIs aus der CSV-Datei zu laden
def load_embeddings(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['CUI'] + [f'feature_{i}' for i in range(1, 51)]
    embeddings = df.drop('CUI', axis=1).values
    return df, embeddings

# Funktion, um die Embeddings zu clustern
def cluster_embeddings(embeddings, n_clusters=100):
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False)
    kmeans.train(embeddings.astype('float32'))
    _, cluster_assignments = kmeans.index.search(embeddings.astype('float32'), 1)
    return kmeans, cluster_assignments

# Funktion, um ähnliche CUIs in einem Cluster zu finden
def find_similar_cuis_in_cluster(target_cui, embeddings, df, cluster_assignments, top_n=5):
    target_index = df.index[df['CUI'] == target_cui].tolist()[0]
    target_cluster = cluster_assignments[target_index][0]
    indices_in_cluster = np.where(cluster_assignments == target_cluster)[0]
    
    index_cluster = faiss.IndexFlatL2(embeddings.shape[1])
    embeddings_cluster = embeddings[indices_in_cluster].astype('float32')
    index_cluster.add(embeddings_cluster)
    
    target_vector = embeddings[target_index:target_index+1].astype('float32')
    distances, indices = index_cluster.search(target_vector, top_n + 1)
    
    similar_cuis = [(df.iloc[indices_in_cluster[i]]['CUI'], 1 - distances[0][j]) for j, i in enumerate(indices[0]) if indices_in_cluster[i] != target_index]
    
    return similar_cuis[:top_n]

# Laden der Embeddings
csv_file_path = input("Geben Sie den Pfad zur CSV-Datei ein: ")  # Pfad zur CSV-Datei
df, embeddings = load_embeddings(csv_file_path)

# Clustern der Embeddings
kmeans, cluster_assignments = cluster_embeddings(embeddings)

# Eingabe durch den Benutzer
target_cui = input("Geben Sie das Ziel-CUI ein: ")
top_n = int(input("Wie viele ähnliche CUIs möchten Sie anzeigen? "))

# Finden und Anzeigen ähnlicher CUIs
similar_cuis = find_similar_cuis_in_cluster(target_cui, embeddings, df, cluster_assignments, top_n)
print(f"Top {top_n} ähnliche CUIs zu {target_cui}:")
for cui, similarity in similar_cuis:
    print(f"{cui}: {similarity:.4f}")
