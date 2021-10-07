''' proof of concept: regression by classification
'''


import os
import random
import numpy as np

with open('data/penstate/data_16k.lst', 'r') as f:
    lines = f.readlines()

dataset = []
for line in lines:
    pid, gender, ancestry, voice_path, face_path = line.rstrip().split(' ')
    item = {'pid': pid,
            'gender': gender,
            'ancestry': ancestry,
            'face_path': face_path,
    }
    dataset.append(item)


key = random.choice(['gender', 'ancestry'])
split = 700
random.shuffle(dataset)
train_set = dataset[:split]
test_set = dataset[split:]
while len({item[key] for item in train_set}) != len({item[key] for item in dataset}):
    random.shuffle(dataset)
    train_set = dataset[:split]
    test_set = dataset[split:]


for item in train_set + test_set:
    face_path = item['face_path']
    face = np.loadtxt(face_path, dtype=np.float32)
    item['face'] = face


train_faces = [item['face'] for item in train_set]
train_faces = np.array(train_faces, dtype=np.float32)
face_mean = np.mean(train_faces, axis=0, keepdims=False)
face_delta = train_faces - np.expand_dims(face_mean, axis=0)
face_dist = np.sum(np.square(face_delta), axis=2, keepdims=True)
face_std = np.sqrt(np.mean(face_dist, axis=0, keepdims=False))

for item in train_set + test_set:
    item['face'] = (item['face'] - face_mean) / face_std


c = 0
avg_face = 0.
count = {}
face_template = {}
for item in train_set:
    face = item['face']
    c += 1
    avg_face += face

    covar = item[key]
    if covar not in face_template:
        face_template[covar] = 0.
    face_template[covar] += face

    if covar not in count:
        count[covar] = 0.
    count[covar] += 1

# 
avg_face /= c
for covar in face_template:
    face_template[covar] /= count[covar]



'''error
'''
avg_error = 0.
covar_error = 0.
err_map = 0.
for item in test_set:
    face = item['face']

    avg_error += np.sum(np.square(avg_face - face), axis=1).mean()

    covar = item[key]
    covar_face = face_template[covar]
    err = np.sum(np.square(covar_face - face), axis=1)
    covar_error += err.mean()
    err_map += err

np.savetxt('err_map_4.txt', err_map / len(test_set), fmt='%.4f')
print(err_map.mean() / len(test_set))


print(key, avg_error / len(test_set),
        covar_error / len(test_set),
        covar_error / avg_error)

