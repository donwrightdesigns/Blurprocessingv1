import sys
import argparse
import logging
import pathlib
import json
import shutil # Added for moving files

import cv2

from blur_detection import estimate_blur
from blur_detection import fix_image_size
from blur_detection import pretty_blur_map


def parse_args():
    parser = argparse.ArgumentParser(description='Run blur detection on images and optionally move blurry ones.')
    parser.add_argument('-i', '--images', type=str, nargs='+', required=True, help='Directory or list of image files/directories.')
    parser.add_argument('-s', '--save-path', type=str, default=None, help='Path to save output JSON results (e.g., results.json).')

    parser.add_argument('-t', '--threshold', type=float, default=100.0, help='Blurry threshold score (lower means more likely to be blurry).')
    parser.add_argument('-f', '--variable-size', action='store_true', help='Do NOT fix the image size before processing (default is to fix size).')

    parser.add_argument('-v', '--verbose', action='store_true', help='Set logging level to DEBUG.')
    parser.add_argument('-d', '--display', action='store_true', help='Display images and their blur maps (press any key to continue, \'q\' to quit).')
    
    # New arguments for moving blurry images
    parser.add_argument('--move-blurry', action='store_true', help='If set, move blurry images to a subdirectory.')
    parser.add_argument('--blurry-subdir', type=str, default='blurry_images', help='Name of the subdirectory for blurry images (default: blurry_images). Created within each image\'s original parent directory.')

    return parser.parse_args()


def find_images(image_paths, img_extensions=['.jpg', '.png', '.jpeg']):
    img_extensions += [i.upper() for i in img_extensions] # Handle uppercase extensions

    for path_str in image_paths:
        path = pathlib.Path(path_str)

        if path.is_file():
            if path.suffix not in img_extensions:
                logging.info(f'{path.suffix} is not a recognized image extension! Skipping {path}')
                continue
            else:
                yield path
        elif path.is_dir():
            for img_ext in img_extensions:
                for img_path in path.glob(f'*{img_ext}'):
                    yield img_path
        else:
            logging.warning(f"Provided path '{path}' is neither a file nor a directory. Skipping.")


def main():
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(module)s : %(message)s', # Added module for clarity
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
    )

    fix_size = not args.variable_size # True if we should fix size, False if variable-size flag is set
    logging.info(f'Fix image size before processing: {fix_size}')
    logging.info(f'Blurry threshold: {args.threshold}')
    if args.move_blurry:
        logging.info(f'Moving blurry images to subdirectory named: "{args.blurry_subdir}"')


    if args.save_path is not None:
        save_path = pathlib.Path(args.save_path)
        if save_path.suffix.lower() != '.json': # More robust check
            logging.warning(f"Specified save_path '{save_path}' does not end with .json. Appending .json")
            save_path = save_path.with_suffix('.json')
    else:
        save_path = None

    results = []
    moved_count = 0
    error_moving_count = 0
    processed_image_count = 0

    for image_path in find_images(args.images):
        processed_image_count +=1
        logging.debug(f'Reading image: {image_path}')
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f'Failed to read image from {image_path}; skipping!')
            continue

        logging.info(f'Processing {image_path}')

        if fix_size:
            processed_image_for_blur = fix_image_size(image.copy()) # Operate on a copy for safety
        else:
            logging.warning('Not normalizing image size for consistent scoring! Scores might vary based on image dimensions.')
            processed_image_for_blur = image.copy()

        blur_map, score, blurry = estimate_blur(processed_image_for_blur, threshold=args.threshold)

        logging.info(f'Image: {image_path.name}, Score: {score:.2f}, Blurry: {blurry}')
        results.append({'input_path': str(image_path), 'score': score, 'blurry': blurry})

        # --- Logic to move blurry images ---
        if args.move_blurry and blurry:
            try:
                original_image_dir = image_path.parent
                target_blurry_dir = original_image_dir / args.blurry_subdir
                
                target_blurry_dir.mkdir(parents=True, exist_ok=True)
                
                destination_path = target_blurry_dir / image_path.name
                
                if destination_path.exists():
                    logging.warning(f"File '{destination_path}' already exists. Skipping move for '{image_path}'.")
                elif image_path.exists(): 
                    shutil.move(str(image_path), str(destination_path))
                    logging.info(f"MOVED blurry image: '{image_path}' -> '{destination_path}'")
                    moved_count +=1
                else:
                    # This case might happen if the image was already moved by another process or in a previous step
                    # of a more complex pipeline if this script were part of one.
                    logging.warning(f"Source image '{image_path}' not found for moving. It might have been moved or deleted.")

            except Exception as e:
                logging.error(f"Error moving blurry image '{image_path}' to '{destination_path}': {e}")
                error_moving_count += 1
        # --- End of move logic ---

        if args.display:
            display_image = image # Show original image for context
            if fix_size and image.shape != processed_image_for_blur.shape: # If size was fixed, it's good to see what was processed
                # Optionally, resize original to match processed_image_for_blur for display if sizes differ significantly
                # For now, just show original. You might want to show 'processed_image_for_blur' too.
                 pass


            cv2.imshow('Input Image', display_image)
            cv2.imshow('Blur Map (Processed)', pretty_blur_map(blur_map))

            key = cv2.waitKey(0) # Wait indefinitely for a key press
            if key == ord('q'):
                logging.info('Exiting due to "q" key press...')
                cv2.destroyAllWindows()
                break # Exit the loop
            cv2.destroyAllWindows() # Close windows after each image display for simplicity

    if save_path is not None:
        logging.info(f'Saving JSON results to {save_path}')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "source_image_paths_or_dirs": args.images,
            "blur_threshold_used": args.threshold,
            "image_size_fixed_for_scoring": fix_size,
            "results": results
        }
        if args.move_blurry:
            output_data["blurry_images_intended_subdir"] = args.blurry_subdir
            output_data["blurry_images_successfully_moved"] = moved_count
            if error_moving_count > 0:
                output_data["blurry_images_move_errors"] = error_moving_count

        try:
            with open(save_path, 'w') as results_file:
                json.dump(output_data, results_file, indent=4)
            logging.info(f"Successfully saved results to {save_path}")
        except IOError as e:
            logging.error(f"Could not write results to {save_path}: {e}")
    
    logging.info(f"Processing complete. Total images considered: {processed_image_count}.")
    if args.move_blurry:
        logging.info(f"Total blurry images moved: {moved_count}")
        if error_moving_count > 0:
            logging.warning(f"Errors encountered while trying to move images: {error_moving_count}")

if __name__ == '__main__':
    main()