import pickle
import numpy as np
from collections import defaultdict, Counter

pickle_file = 'itl.p'
itl = pickle.load(open(pickle_file, "rb"))

songf = pickle.load(open("songsf.p", 'rb'))

song_list = []
count = 0
itl_song_keys = list(itl.songs.keys())
#print(itl_song_keys)
for key, value in songf.items():
	while count < len(itl_song_keys):
		if int(itl_song_keys[count]) == int(key):
			song_list.append((int(key), str(itl.songs[int(key)].album)))
			count += 1
			break
		else:
			count += 1

song_list = sorted(song_list, key=lambda x: x[1])

int_to_key = list()
int_to_key.append(-1)

key_to_int = dict()
key_to_int[-1] = 0
int_val = 1

for key, album in song_list:
	key_to_int[key] = int_val
	int_to_key.append(key)
	int_val += 1

partitioned_list = []
lastAlbum = None
count = -1

for key, album in song_list:
	if album == 'Voice Memos':
		continue
	if album == lastAlbum:
		partitioned_list[count].append(key_to_int[key])
	else:
		count += 1
		partitioned_list.append([0])
		partitioned_list[count].append(key_to_int[key])
		lastAlbum = album

print(partitioned_list)
pickle.dump({'itk': int_to_key, 'kti': key_to_int, 'ixx': partitioned_list}, open('input.p', 'wb'), protocol=2)