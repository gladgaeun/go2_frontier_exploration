# Image-text relevance of one camera image (TravExplorer Sec. V-B: BLIP-2 image-text matching),
# stamped like the RGB / depth image it was computed on.
std_msgs/Header header        # stamp and frame of the camera image
string text                   # the text it was matched against (e.g. "Seems like there is a bed ahead.")
float32 score                 # relevance s^ in [0, 1]
