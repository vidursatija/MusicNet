from libpytunes import Library
import librosa
import numpy as np
#import matplotlib.pyplot as plt
#import time
import pickle
import os

def calculateFeatures(filename):
	y, sr = librosa.load(filename)
	hop_length = 512
	section_length = 30
	permit_length = 0 #No overlap
	n_paras = 4

	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
	n_beats = len(beat_frames)
	y_harmonic, _ = librosa.effects.hpss(y)
	beat_times = librosa.frames_to_time(beat_frames, sr=sr)
	mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=1)
	mfcc_delta = librosa.feature.delta(mfcc)
	beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

	chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
	beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

	delta_rms = np.square(beat_mfcc_delta[1])

	prev_delta_sums = np.empty((n_beats-section_length+1))
	total_sum = np.sum(delta_rms[:section_length])
	prev_delta_sums[0] = total_sum
	for pos in range(n_beats-section_length):
		total_sum = total_sum - delta_rms[pos] + delta_rms[pos+section_length]
		prev_delta_sums[pos+1] = total_sum

	prev_delta_sums_delta = librosa.feature.delta(prev_delta_sums)
	para_init_locs = []
	for n_p in range(n_paras):
		lowest = 50
		lowest_loc = 0
		for loc, each_sum_delta in enumerate(prev_delta_sums_delta):
			#Check valid loc
			valid_loc = True
			for each_loc in para_init_locs:
				if loc > each_loc - (section_length - permit_length) and loc < each_loc + (section_length - permit_length):
					valid_loc = False
					break

			if each_sum_delta <= lowest and valid_loc:
				lowest = each_sum_delta
				lowest_loc = loc

		para_init_locs.append(lowest_loc)

	para_init_locs.sort()
	#print(para_init_locs)

	all_features = np.empty((n_paras*section_length, 2)) #0 - mfcc, 1...3 - chroma
	for n_p in range(n_paras):
		all_features[n_p*section_length:(n_p+1)*section_length, 0] = beat_mfcc_delta[0][para_init_locs[n_p]:para_init_locs[n_p]+section_length] / 250
		all_features[n_p*section_length:(n_p+1)*section_length, 1] = np.argmax(beat_chroma[:, para_init_locs[n_p]:para_init_locs[n_p]+section_length], axis=0)/11

	return all_features.reshape((n_paras*section_length*2))

pickle_file = "itl.p"
xmlPath = '/Users/vidursatija/Music/iTunes/iTunes Music Library.xml' #your dir here
l = Library(xmlPath)
pickle.dump(l, open(pickle_file, "wb"))

itl = pickle.load(open(pickle_file, "rb"))

song_dict = {}
count = 0
for id, song in itl.songs.items():
	if song and song.kind:
		if song.kind[-10:] == 'audio file':
			#songPath = '/Users/vidursatija/Music/iTunes/iTunes Media/Music/The 1975/The 1975/11 Girls.mp3'
			songPath = song.location
			try:
				song_dict[str(id)] = calculateFeatures(os.path.join('/', songPath))
				count += 1
			except Exception as e:
				print("Booyeah!")
	if count % 5 == 1:
		print(count)

print(count)
pickle.dump(song_dict, open('songsf.p', 'wb'), protocol=2)


#time1 = time.time()
#time2 = time.time()
#print(float(time2-time1))

