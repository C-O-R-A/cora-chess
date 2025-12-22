import os
import sys
sys.path.append(os.path.abspath(".."))
import numpy as np
import chess
from tfs import transforms as tf

def test_TF_and_get_vec_from_TF():
    pos = np.array([0.1, 0.2, 0.3])
    rot = np.array([0.1, 0.2, 0.3])
    
    TF_matrix = tf.TF(pos, rot)
    extracted_pos, extracted_rot = tf.get_vec_from_TF(TF_matrix)
    
    assert np.allclose(pos, extracted_pos), "Position extraction failed"
    assert np.allclose(rot, extracted_rot), "Rotation extraction failed"