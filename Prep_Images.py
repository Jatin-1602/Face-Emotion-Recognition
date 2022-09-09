import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

# emotions = np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))

# Making Folders
train_test = ['train', 'test']
emotion_dir = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)

for folder in train_test:
    os.makedirs(os.path.join('data', folder), exist_ok=True)
    for emotion_folder in emotion_dir:
        os.makedirs(os.path.join('data', folder, emotion_folder), exist_ok=True)


# keep count of each category
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0


# Reading CSV file
df = pd.read_csv("fer2013.csv/fer2013.csv")
img_matrix = np.zeros((48, 48), dtype=np.uint8)

path_train = 'data/train/'
path_test = 'data/test/'

print("Saving images...")

for i in tqdm(range(len(df))):
    line = df['pixels'][i]
    img_pixels = line.split(' ')

    # img_size = 48 x 48 = 2304 (len(line))
    for j in range(len(img_pixels)):
        x = j // 48
        y = j % 48
        img_matrix[x][y] = int(img_pixels[j])

    # converts a numerical (integer or float) numpy array
    # of any size and dimensionality into a CASA image

    img = Image.fromarray(img_matrix)

    emotion_index = df['emotion'][i]

    # 80% of len(df) == train
    if (i < 28709):
        if (emotion_index == 0):
            img.save(path_train + "angry/" + str(angry+1) + '.png')
            angry += 1
        elif emotion_index == 1:
            img.save(path_train + "disgusted/" + str(disgusted + 1) + '.png')
            disgusted += 1
        elif emotion_index == 2:
            img.save(path_train + "fearful/" + str(fearful + 1) + '.png')
            fearful += 1
        elif emotion_index == 3:
            img.save(path_train + "happy/" + str(happy + 1) + '.png')
            happy += 1
        elif emotion_index == 4:
            img.save(path_train + "sad/" + str(sad + 1) + '.png')
            sad += 1
        elif emotion_index == 5:
            img.save(path_train + "surprised/" + str(surprised + 1) + '.png')
            surprised += 1
        elif emotion_index == 6:
            img.save(path_train + "neutral/" + str(neutral + 1) + '.png')
            neutral += 1

    # 20% of len(df) == test
    else:
        if (emotion_index == 0):
            img.save(path_test + "angry/" + str(angry_test+1) + '.png')
            angry_test += 1
        elif emotion_index == 1:
            img.save(path_test + "disgusted/" + str(disgusted_test + 1) + '.png')
            disgusted_test += 1
        elif emotion_index == 2:
            img.save(path_test + "fearful/" + str(fearful_test + 1) + '.png')
            fearful_test += 1
        elif emotion_index == 3:
            img.save(path_test + "happy/" + str(happy_test + 1) + '.png')
            happy_test += 1
        elif emotion_index == 4:
            img.save(path_test + "sad/" + str(sad_test + 1) + '.png')
            sad_test += 1
        elif emotion_index == 5:
            img.save(path_test + "surprised/" + str(surprised_test + 1) + '.png')
            surprised_test += 1
        elif emotion_index == 6:
            img.save(path_test + "neutral/" + str(neutral_test + 1) + '.png')
            neutral_test += 1

    # To view the Image
    # from matplotlib.pyplot import imshow
    # imshow(img)

print("Done!!!")