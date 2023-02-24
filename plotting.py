import numpy as np
from PIL import Image
import arabic_reshaper
import matplotlib.pyplot as plt
from bidi.algorithm import get_display

def plot_heatmap(result_matrix):
  height, width = result_matrix.shape
  fig, ax = plt.subplots()
  fig.set_size_inches(8,8)
  im = ax.imshow(result_matrix)
  # Create X & Y Labels
  ax.set_xticks(np.arange(width))
  ax.set_yticks(np.arange(height))
  ax.set_xticklabels(["Image {}".format(i) for i in range(width)])
  ax.set_yticklabels(["Text {}".format(i) for i in range(height)])

  for i in range(height):
    for j in range(width):
        text = ax.text(j, i, result_matrix[i, j],
                       ha="center", va="center", color='grey', size=20)

  fig.tight_layout()
  plt.show()


def display_image_and_caption(image_id, caption_en = [], capions_ar = []):
    image_path = 'data/Images/' + image_id
    image = Image.open(image_path)
    plt.imshow(image)
    caption = ''
    for c in caption_en:
        caption += c + '\n'
    for c in capions_ar:
        c = get_display(arabic_reshaper.reshape(c))
        caption += c + '\n'
    plt.title(caption)
    plt.axis('off')
    plt.show()