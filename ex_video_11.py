from scipy import datasets
import matplotlib.pyplot as plt

face = datasets.face(gray=True)
h, w = face.shape
print(w, h)
face = face[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]

# Bonus
face[face >= 128] = 255
face[face < 128] = 0

plt.imshow(face, cmap=plt.cm.gray)
plt.show()
