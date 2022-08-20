import numpy as np
from typing import Dict, List
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngImageFile

def get_pil_image_grid(images: List[PngImageFile], headers: List[str] = None) -> PngImageFile:

    assert all([images[i].size == images[i+1].size for i in range(len(images)-1)])
    
    w, h = images[0].size
    rows = cols = int(np.ceil(np.sqrt(len(images))))

    if headers is None: 
        headers = [None] * len(images)

    grid_image = Image.new('RGB', size=(cols*(w+1), rows*(h+1)))

    new_images = []
    for image in images:
        new_image = Image.new('RGB', size=(w+1, h+1))
        new_image.paste(image, box=(0,0))
        new_images.append(new_image)

    w, h = w+1, h+1
    
    for i, (image, header) in enumerate(zip(new_images, headers)):
        if header:
            draw = ImageDraw.Draw(image)
            draw.text((0, 0),header,(0, 0, 0))
        grid_image.paste(image, box=(i%cols*w, i//cols*h))

    return grid_image

def save_pil_image(image: PngImageFile, name: str):
    image.save(f'{name}.jpg')

def show_pil_image(image: PngImageFile):
    image.show()

def get_pil_image_metadata(image: PngImageFile) -> Dict:
    metadata_dict = {
        "File Name": image.filename,
        "Image Size": image.size,
        "Image Height": image.height,
        "Image Width": image.width,
        "Image Format": image.format,
        "Image Mode": image.mode,
        "Image is Animated": getattr(image, "is_animated", False),
        "Frames in Image": getattr(image, "n_frames", 1),
        "Frames in Image": getattr(image, "n_frames", 1)
    }

    exifdata = image.getexif()

    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
            data = data.decode()
        metadata_dict[tag] = data
    
    return metadata_dict

