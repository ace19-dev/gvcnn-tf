# TODO:
#
# In GVCNN, the shape descriptor comes
# from the output of group fusion module, which is more rep-
# resentative than the view descriptor extracted from single
# view. And we directly use it for 3D shape retrieval.
# For two 3D shape X and Y , x and y is the shape descriptor
# extracted from GVCNN. Concretely, we use Euclidean dis-
# tance between two 3D shapes in retrieval.

# We further adopt a low-rank Mahalanobis metric. We learn
# a Mahalanobis metric W that directly projects GVCNN de-
# scriptors to a new space, in which the intra-class distance is
# smaller and inter-class distance is larger. We use the large-
# margin metric learning algorithm and implementation from