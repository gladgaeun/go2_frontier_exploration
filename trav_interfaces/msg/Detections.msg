# Open-vocabulary detections of one camera image (TravExplorer Sec. V-A): one instance mask and
# confidence per detection d_q = (R_q, s_q, q), registered with the RGB / depth image of the stamp.
std_msgs/Header header        # stamp and frame of the camera image
uint32 height
uint32 width
string[] queries              # query q of each detection (e.g. "stairs", "couch")
float32[] scores              # detection confidence s_q
uint8[] masks                 # region R_q of each detection: one bit per pixel, row-major,
                              # ceil(height * width / 8) bytes per detection (numpy.packbits order)
