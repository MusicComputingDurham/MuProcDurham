#!/usr/bin/env python
# coding: utf-8
from tests.notebooks.util import get_ipython

# ## Practical: Rhythms and Scores
# 
# By the end of this practical you should able to:
# 
# - use `music21` to produce score representations of simple melodies
# - create and listen to some rhythms with interesting properties
# - use a simple counting approach to distinguish between various metrical prototypes 
# 
# This notebook is for non-commercial use only.

# ## 1. Representing and typesetting melodies with `music21`

# `music21` is a Python module that has been developed over the last 15 years or so, and can be useful for quick manipulations of score-based symbolic music representations, where information conveyed only by staff notation  (e.g. key signatures, spelling, barlines etc.) is required for your application.
# 
# It is quite idiosyncratic, so we will only touch on it briefly in order to demonstrate the kinds of things a fully fledged score-based symbolic music representation should be able to do. 

# In[1]:


from music21 import environment, stream, note, converter, corpus
from IPython.display import Image, Audio

import music21
import pretty_midi
import numpy as np


# `music21` depends on several optional external binaries to support a variety of output formats. Here, we depend on a programme called GNU Lilypond, which is a music typesetting engine (think $\LaTeX$ for music). On NCC, `lilypond` lives at `/usr/bin/lilypond`. This may be different on your computer.
# 

# In[2]:


environment.UserSettings()['lilypondPath'] = '/usr/bin/lilypond'


# Please note that the support that `music21` has for `lilypond` is extremely limited; we just use it as an output format here because it produces images that do not have any other dependencies. You can use this convenience function (`show_object`) to typeset many `music21` objects.

# In[3]:


def show_object(music21_object):
    lpc = music21.lily.translate.LilypondConverter()
    lpc.loadFromMusic21Object(music21_object)
    path = lpc.createPNG()
    display(Image(filename=lpc.createPNG()))
    return path


# Consider the following melody
# 
# ![Simple score 1](./melody1.png)

# We can construct this in `music21` as follows:

# In[4]:


s = music21.stream.Stream()
s.append(music21.key.Key('E-'))
s.append(music21.meter.TimeSignature('2/4'))

s.append(music21.note.Rest(quarterLength=0.5))
s.append(music21.note.Note('G4', quarterLength=0.5))
s.append(music21.note.Note('G4', quarterLength=0.5))
s.append(music21.note.Note('G4', quarterLength=0.5))
s.append(music21.note.Note('E4-', quarterLength=2))


# Use `show_object` to typeset the `music21` `Stream` instance (it works with many types of `music21` object)

# In[5]:


path = show_object(s)


# ### 1A 
# 
# Create a `music21.stream.Stream()` that matches this score:
# 
# ![Simple score 1A](melody4.png)

# In[ ]:


q1a = music21.stream.Stream()
# vvvvvvvvvvvvvvvv
q1a.append(music21.key.Key('C'))
q1a.append(music21.meter.TimeSignature('3/4'))

for spn, ql in zip(['C5', 'D5', 'E5', 'F5', 'G5', 'A5'], range(1, 6)):
    q1a.append(music21.note.Note(spn, quarterLength=ql/4))
# ^^^^^^^^^^^^^^^^^
show_object(q1a)


# ### 1B
# 
# Create a `music21.stream.Stream()` that matches this score:
# 
# ![Simple score 1B](./melody2.png)

# In[ ]:


q1b = music21.stream.Stream()
# vvvvvvvvvvvvvvvv
q1b.append(music21.key.Key('F'))
q1b.append(music21.meter.TimeSignature('2/4'))

q1b.append(music21.note.Rest(quarterLength=0.5))
q1b.append(music21.note.Note('A4', quarterLength=0.25))
q1b.append(music21.note.Note('B4', quarterLength=0.25))
q1b.append(music21.note.Note('C5', quarterLength=0.75))
q1b.append(music21.note.Note('A4', quarterLength=0.25))
q1b.append(music21.note.Note('G4', quarterLength=0.5))
q1b.append(music21.note.Note('F4', quarterLength=1))
# ^^^^^^^^^^^^^^^^^
show_object(q1b)


# ### 1C
# 
# Create a `music21.stream.Stream()` that matches this score:
# 
# ![Simple score 1A](./melody3.png)

# In[ ]:


import copy 

q1c = music21.stream.Stream()
# vvvvvvvvvvvvvvvv
q1c.append(music21.key.Key('D'))
q1c.append(music21.meter.TimeSignature('6/8'))

pattern = [
    music21.note.Rest(quarterLength=0.5),
    music21.note.Note('D4', quarterLength=0.5),
    music21.note.Note('E4', quarterLength=0.5)
]

for _ in range(5):
    for element in pattern:
        q1c.append(copy.deepcopy(element))

q1c.append(music21.note.Note('F#4', quarterLength=1))
q1c.append(music21.note.Note('G4', quarterLength=0.5))
q1c.append(music21.note.Note('A4', quarterLength=1))
# ^^^^^^^^^^^^^^^^^


# In[9]:


show_object(q1c)


# ## Interlude. Common metrical patterns for a time signature

# In this section, we look at how the time signature of piece of music relates to the kinds of rhythms that appear in it. To do this, we will examine rhythmic patterns in a large collection of folk songs, called the Essen corpus.

# In[10]:


import collections
import multiprocessing 

from matplotlib import pylab as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

def process_score(ep):
    data = collections.defaultdict(list)
    a = corpus.parse(ep)
    for s in a.scores:
        piece_start_times = []
        tses = list(s.flat.getElementsByClass('TimeSignature'))
        parts = list(s.parts)
        if len(tses) == 1 and len(parts) == 1:
            sig = tses[0].ratioString
            for m in s.parts[0].getElementsByClass('Measure'):
                measure_duration = m.duration.quarterLength
                measure_start_times = [note.offset / measure_duration for note in m.notes]
                piece_start_times.extend(measure_start_times)
            data[sig].append(piece_start_times)
    return data

def produce_rhythm_data():
    essen_pieces = corpus.getComposer('essenFolksong')
    num_cores = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_score, essen_pieces)
    
    combined_data = collections.defaultdict(list)
    for result in results:
        for key, value in result.items():
            combined_data[key].extend(value)
    
    return combined_data

# takes 3m 7.6s on 8 cores in WSL2 
data = produce_rhythm_data()


# In[11]:


all_features = []
for piece_counts in data['3/4']:
    if piece_counts != []:
        features, bins = np.histogram(piece_counts, bins=128, density=True)
        all_features.append(features)

X = np.array(all_features)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7, edgecolors='b')
plt.title('PCA of measure weight features')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# ## 3. Considerations when converting from score to MIDI

# Consider the following melody again
# 
# ![Simple score 1](./melody1.png)

# The `music21` object can be converted to a series of `pretty_midi` `Notes` with onset times and durations. Hower, because `music21` is a score-based representation and `pretty_midi` deals with musical time in terms of seconds (which is better than "tick" time), we need to apply our understanding of tempo to figure out the start and end times of the notes.

# In[ ]:


def m21_to_pm(m21_stream, bpm=120):
    pm = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=42)

    # beat_to_quarter is the ratio between beats and quarter notes
    beat_to_quarter = m21_stream.timeSignature.beatLengthToQuarterLengthRatio
    # beat_count is the number of beats in a measure (bar)
    beat_count = m21_stream.timeSignature.beatCount

    # vvvvvvvvvvvvvvvv
    # quarter_note_duration is the duration of a quarter note in seconds
    quarter_note_duration = 60 / bpm
    # ^^^^^^^^^^^^^^^^

    for note in m21_stream.notes:
        # offset is the start time of the note, measured in quarter notes
        # quarterLength is the duration of the note, measured in quarter notes
        offset, quarterLength = note.offset, note.quarterLength

        # vvvvvvvvvvvvvvvv
        start_time = offset * quarter_note_duration
        duration = quarterLength * quarter_note_duration
        end_time = start_time + duration

        pm_note = pretty_midi.Note(
            velocity=note.volume.velocity,
            pitch=note.pitch.midi,
            start=start_time,
            end=end_time
        )
        instrument.notes.append(pm_note)
        # ^^^^^^^^^^^^^^^^
    
    pm.instruments.append(instrument)

    return pm


# ## 4. A canon in augmentation by Josquin des Prez

# We can define a transformation on melodies, called "augmentation", which increases the duration of every note by a constant factor. Write a function `augment(note_list, factor)` which augments a melody represented in `note_list`, which is a list of `pretty_midi` `Note`s by an arbitrary factor `factor`.

# In[ ]:


def rhythm_augment(note_list, factor):
    augmented_notes = []
    for note in note_list:

        # vvvvvvvvvvvvvvvv
        new_start = note.start * factor
        new_end = note.end * factor

        augmented_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=new_start,
            end=new_end
        )
        
        augmented_notes.append(augmented_note)
        # ^^^^^^^^^^^^^^^^
    return augmented_notes


# Now let's listen to the effect of combining these three augemented versions together:

# In[14]:


v1 = rhythm_augment(pretty_midi.PrettyMIDI('josquin.midi').instruments[0].notes, 1)
v2 = rhythm_augment(pretty_midi.PrettyMIDI('josquin.midi').instruments[0].notes, 3)
v3 = rhythm_augment(pretty_midi.PrettyMIDI('josquin.midi').instruments[0].notes, 3/2)


# In[15]:


prolation = pretty_midi.PrettyMIDI()

i1 = pretty_midi.Instrument(program=1)
i2 = pretty_midi.Instrument(program=2)
i3 = pretty_midi.Instrument(program=3)

i1.notes = v1

for note in v2:
    note.pitch -= 5 

i2.notes = v2

for note in v3:
    note.pitch -= 12

i3.notes = v3

prolation.instruments = [i1, i2, i3]


# In[16]:


Audio(prolation.synthesize(fs=22050), rate=22050)
