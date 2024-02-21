
README: Similar CUI Finder
Overview
This Python Script provides a straightforward and interactive way to find semantically similar Concept Unique Identifiers (CUIs) from the Unified Medical Language System (UMLS) based on precomputed embeddings. Utilizing a combination of Python, Pandas for data handling, Faiss for efficient similarity search, and t-SNE for dimensionality reduction, it offers an efficient tool for exploring relationships between medical concepts.

How It Works
Loading Embeddings: The script begins by loading embeddings from a CSV file, where each row represents a medical concept (CUI) and its associated vector representation in a high-dimensional space. Download Embeddings from this repository: https://github.com/r-mal/umls-embeddings

Clustering Embeddings: To improve search efficiency, embeddings are clustered using Faiss's implementation of the k-means algorithm. This step groups similar embeddings together, reducing the search space for finding related CUIs.

Finding Similar CUIs: Users can input a target CUI and specify the number of similar CUIs they wish to retrieve. The script then identifies the cluster containing the target CUI and performs a similarity search within that cluster to find the top-n most similar CUIs.










