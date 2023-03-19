
from PIL import Image
import sys
import time
import numpy as np
import NDIlib as ndi
from pathlib import Path


def main():

    if not ndi.initialize():
        return 0

    ndi_send = ndi.send_create()

    if ndi_send is None:
        return 0
    
    video_frame = ndi.VideoFrameV2()

    idx = 1
    while True:
        # every 60 frames print the fps
        if idx % 60 == 0:
            fidx = int(idx/60)
            print(f"Loading frame {fidx:06d}.png")
           
            # if the image path exists
            if Path(f"frames/{fidx:06d}.png").is_file():
                # load image frame in folder frames for idx
                try:
                    img_pil = Image.open(f"frames/{fidx:06d}.png")
                    img_pil = img_pil.convert('RGBA')
                    img_pil = img_pil.resize((1280, 720))
                except:
                    idx -= 60
            else:
                idx -= 60
                #update the content of image with a img_pil as array
            video_frame = ndi.VideoFrameV2()
            img = np.array(img_pil)
            video_frame.data = img
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA

        ndi.send_send_video_v2(ndi_send, video_frame)
        idx += 1

    ndi.send_destroy(ndi_send)

    ndi.destroy()

    return 0

if __name__ == "__main__":
    sys.exit(main())
