all: kmeans agglomerative dbscan

.PHONY: kmeans agglomerative dbscan

kmeans:
	python kmeans/main.py

agglomerative:
	python agglomerative/main.py

dbscan:
	python dbscan/dbscan.py
