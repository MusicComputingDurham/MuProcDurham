import re
import random
from warnings import warn

import numpy as np
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # supress pygame message before import
import pygame
import music21 as m21

# check for most up-to-date version
import muprocdurham.version_check

# see https://github.com/DCMLab/standards/blob/main/harmony.py
dcml_harmony_regex = re.compile("".join(r"""
    ^(\.?
        ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
        ((?P<localkey>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
        ((?P<pedal>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
        (?P<chord>
            (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
            (?P<form>(%|o|\+|M|\+M))?
            (?P<figbass>(7|65|43|42|2|64|6))?
            (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
            (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
        )
        (?P<pedalend>\])?
    )?
    (|(?P<cadence>((HC|PAC|IAC|DC|EC)(\..*?)?)))?
    (?P<phraseend>(\\\\|\{|\}|\}\{))?$
    """.split()))


def seed_everything(seed):
    """Set random seed for reproducibility across multiple libraries."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module


def show_stream(stream):
    try:
        stream.show()
    except m21.converter.subConverters.SubConverterException:
        warn("Cannot show score, falling back to text representation.", UserWarning)
        stream.show('t')


def play_stream(stream):
    try:
        m21.midi.realtime.StreamPlayer(stream).play()
    except pygame.error:
        warn("Cannot play stream", UserWarning)
