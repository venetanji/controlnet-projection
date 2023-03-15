
from PIL import Image
import sys
import time
import numpy as np
import NDIlib as ndi


def main():

    if not ndi.initialize():
        return 0

    ndi_send = ndi.send_create()

    if ndi_send is None:
        return 0
    
    global img 
    img = Image.open("frames/000001.png")

    # convert to RGBA
    img = img.convert('RGBA')
    # resize image to 1280x720
    img = img.resize((1280, 720))

    img = np.array(img)
 
    video_frame = ndi.VideoFrameV2()

    video_frame.data = img
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA

    start = time.time()
    idx = 1
    while True:
        
        start_send = time.time()


            # color = int(255 * idx / 200)
            # img.fill(color if idx % 2 else 0)
        
        
        # every 60 frames print the fps
        if idx % 300 == 0:
            fidx = int(idx/300)
            print(f"Loading frame {fidx:06d}.png")
            # load image frame in folder frames for idx
            img_pil = Image.open(f"frames/{fidx:06d}.png")
            img_pil = img_pil.convert('RGBA')
            img_pil = img_pil.resize((1280, 720))
            
            #update the content of image with a img_pil as array
            video_frame = ndi.VideoFrameV2()
            img = np.array(img_pil)
            video_frame.data = img
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
            print('60 frames sent, at %1.2ffps' % (60.0 / (time.time() - start_send)))

        ndi.send_send_video_v2(ndi_send, video_frame)
        idx += 1

    ndi.send_destroy(ndi_send)

    ndi.destroy()

    return 0

if __name__ == "__main__":
    sys.exit(main())
