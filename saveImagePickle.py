import os
import sys
import face_recognition as fc

cwd = os.path.dirname(sys.argv[0])
img_path = cwd + "/training/images/face-detetction"

# fc.face_encode_and_save(img_path)
# fc.extract_feature_and_save(img_path)

# fc.extract_histogram_save(img_path)
fc.extract_histogram_save(img_path)
