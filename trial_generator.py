import random

high_energy_clips = [
    "pendulum", "dolphins", "coaster", "dvd", "fish", "rays", "dancer", "wind", "escalator", "fireworks"
]

low_energy_clips = [
    "birds", "cars", "train", "waves", "candle", "puppy", "jellyfish", "aquarium", "blossoms", "blooming"
]

all_clips = high_energy_clips + low_energy_clips


trial_videos = random.sample(high_energy_clips, 5) + random.sample(low_energy_clips, 5)
random.shuffle(trial_videos)

match_indices = random.sample(range(10), 5)
trial_audios = [None] * 10
for i in match_indices:
    trial_audios[i] = trial_videos[i]

for i in range(len(trial_audios)):
    if trial_audios[i] is not None:
        continue
    else:
        mismatched = random.choice(all_clips)
        while mismatched in trial_audios or mismatched == trial_videos[i]:
            mismatched = random.choice(all_clips)
        trial_audios[i] = mismatched

print("VIDEO")
for v in trial_videos:
    print(v)

print()
print("AUDIO")
for v in trial_audios:
    print(v)

for v in all_clips:
    if v not in trial_videos:
        print("example video:", v)
        print("audio", random.choice(["match", "mismatch"]))
        break