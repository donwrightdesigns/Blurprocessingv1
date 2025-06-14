# Blur Detection w optional img SUB-DIRECTORY move
Python script, modified to now cull the images based on results (Add to subdirectories) 

Personally, I use this script for preparing datasets for photogrammetry and Nerf or Gaussian Splat creation

This package only depends on numpy and opencv, to install them run, 

```
pip install -U -r requirements.txt
```

The repository has a script, `process.py` which lets us run on single images or directories of images. The blur detection method is highly dependent on the size of the image being processed. To get consistent scores we fix the image size to HD, to disable this use  `--variable-size`. The script has options to, 

```bash
# run on a single image
python process.py -i input_image.png

# run on a directory of images
python process.py -i input_directory/ 

# or both! 
python process.py -i input_directory/ other_directory/ input_image.png
```

. In addition to logging whether an image is blurry or not, we can also,

```bash
# save this information to json
python process.py -i input_directory/ -s results.json

# display blur-map image
python process.py -i input_directory/ -d
```
The saved json file has information on how blurry an image is, the higher the value, the less blurry the image.

```json
{
    "images": ["/Users/demo_user/Pictures/Flat/"],
    "fix_size": true,
    "results": [
        {
            "blurry": false,
            "input_path": "/Users/demo_user/Pictures/Flat/IMG_1666.JPG",
            "score": 6984.8082115095549
        },
    ],
    "threshold": 100.0
}
```

This is based upon the blogpost [Blur Detection With Opencv](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) by Adrian Rosebrock.


#*NEW* To enable moving blurry images, add the --move-blurry flag:

```bash
python process.py -i s:\test --save-path results_moved.json --move-blurry
```

If you want to specify a different subdirectory name than "blurry_images":

```bash
python process.py -i s:\test --save-path results_moved.json --move-blurry --blurry-subdir "needs_review"
```

