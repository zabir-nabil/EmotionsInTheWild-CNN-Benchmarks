import cv2
import json
import random
import os
import time
import imageio
from PIL import Image, ImageOps


def padding(img, expected_size):
    # https://stackoverflow.com/a/60737369/4622046
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    # https://stackoverflow.com/a/60737369/4622046
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

annotations = json.load(open("annotations.json"))
dataset_root = "emotic"

print(f"Train samples: {len(annotations['train'])}")
print(f"Validation samples: {len(annotations['val'])}")
print(f"Test samples: {len(annotations['test'])}")

gif_frames = []
for random_image in random.sample(annotations['train'], 10):
    frame = cv2.imread(os.path.join(dataset_root, random_image['path']))
    x1, y1, x2, y2 = random_image['bbox']
    label = str(random_image['labels'])
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (36,255,12), 1)
    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99,255,99), 2)

    gif_frames.append(resize_with_padding(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB'), (512, 345)))
    
    cv2.imshow('Emotic Dataset Visualizer', frame)

    if cv2.waitKey(500) == ord('q'):
        # do not close window, you want to show the frame
        cv2.destroyAllWindows()
        break


imageio.mimsave('demo_dataset.gif', gif_frames, duration = 2 )