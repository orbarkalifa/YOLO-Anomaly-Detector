import pickle
import cv2
import numpy as np

routine_map = pickle.load(open('routine_map.pkl', 'rb'))
normalized_map = cv2.normalize(routine_map, None, 0, 255, cv2.NORM_MINMAX)
colored_map = cv2.applyColorMap(normalized_map.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imshow('Routine Map Loaded', colored_map)
cv2.waitKey(0)