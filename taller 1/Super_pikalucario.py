import numpy as np

class MIPCA:
    def __init__(self):
        self.mean = None
        self.components = None
        self.explained_variance = None

    def ajustar(self, X):
        """
        Ajusta el PCA a los datos X
        - Centra los datos
        - Calcula autovalores y autovectores
        - Ordena según varianza explicada
        """
        # 1. Guardar la media
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Calcular matriz de covarianza
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Ordenar de mayor a menor varianza
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_idx]
        self.components = eigenvectors[:, sorted_idx]

    def transformar(self, X, n_componentes):
        """
        Proyecta los datos en n componentes principales
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components[:, :n_componentes])

    def varianza_explicada(self):
        """
        Devuelve la varianza explicada acumulada
        """
        return np.cumsum(self.explained_variance) / np.sum(self.explained_variance)

import numpy as np

class MiKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None
        self.labels_ = None

    def ajustar(self, X):
        # Configurar semilla
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Inicializar centroides aleatoriamente desde los datos
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for i in range(self.max_iter):
            # 1. Asignar cada punto al centroide más cercano
            distancias = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distancias, axis=1)

            # 2. Calcular nuevos centroides
            nuevos_centroides = np.array([X[self.labels_ == j].mean(axis=0) 
                                          for j in range(self.n_clusters)])

            # 3. Calcular inercia (suma de distancias cuadradas al centroide asignado)
            self.inertia_ = np.sum((X - self.centroids[self.labels_])**2)

            # 4. Criterio de parada: cambio de inercia < tol
            if i > 0 and abs(self.inertia_ - prev_inertia) < self.tol:
                print(f"Convergió en {i} iteraciones con tolerancia {self.tol}")
                break

            # Actualizar centroides e inercia previa
            self.centroids = nuevos_centroides
            prev_inertia = self.inertia_

        return self

    def predecir(self, X):
        distancias = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distancias, axis=1)

