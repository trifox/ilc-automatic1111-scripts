import modules.scripts as scripts
import gradio as gr
import os

import math
from PIL import Image, ImageDraw
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

from scripts.ilc_common import getILCBase64ImageSingleton, makePilILCMandelbrotLogo, ILC_LOGGER
ILC_PLUGIN_NAME = "ILC-Image-Stitcher"
ILC_PLUGIN_DESCRIPTION = """
Created to arrange a folder of images, and outpaint the space inbeetween.

This generator tries to use as much as possible of already created images.

Created from the poor outpainting as base, what this script does is to take a folder as input, and arrange each image in the folder in a square manner and fill out the space using outpainting, 
the generated images are stitched together while doing that, having the best interblending of consecutive calls

The images folder, all png and jpg files are read, ideally they are all same dimension, otherwise become cropped to set size
"""
ILC_PLUGIN_VERSION = 1.0
ILC_COMPLETE_NAME = ILC_PLUGIN_NAME+'@'+str(ILC_PLUGIN_VERSION)

LOGGER = ILC_LOGGER(ILC_PLUGIN_NAME, ILC_PLUGIN_VERSION)


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return ILC_COMPLETE_NAME


# Determines when the script should be shown in the dropdown menu via the
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):

        with gr.Blocks():
            with gr.Column(variant="panel"):
                with gr.Row():

                    gr.HTML(
                        """<a href="https://ilovechaos.org" target="chaos">
                        <img style="width:90%; border-radius: 1em; float:left;margin:1em" src="data:image/png;base64, """+getILCBase64ImageSingleton().decode('utf-8')+"""" alt="I-Love-Chaos Logo" />
                    </a>""")

                    gr.Markdown("""
### """+ILC_COMPLETE_NAME+"""

"""+ILC_PLUGIN_DESCRIPTION+"""

this particular plugin is derived and most code reused from
[Animator Script](https://github.com/Animator-Anon/Animator)

We follow a gui first directive here, so we reduce all possible parameters and provide sliders.

Parameter Description
|||
|---|---|
|Folder| A Path on your filesystem containing jpg or png images |
|Item Width/Height| This is the size to what the folder images are scaled to, at best all images in the folder have same resolution |
|Gap Horizontal/Vertical| The gap is applied before and after each item in horizontal vertical manner |
|Items per row| Controls how many columns/lines the arrangement will have, when one row is full the next row will be started  |
|Mask Blur| The gap is applied before and after each item in horizontal vertical manner |
|Masked Content| Controls with what to fill the created space |

"""
                                )
        with gr.Row(variant="panel"):

            with gr.Column(scale=8):
                folder = gr.Textbox(label="Folder",
                                    value='/imagesfolder')
        with gr.Column(variant="panel"):
            item_width = gr.Slider(label="Item Width",
                                   value=512, minimum=16,
                                   maximum=2048, step=16,)
            item_height = gr.Slider(label="Item Height", minimum=16,
                                    maximum=2048, step=16,
                                    value=512)

        with gr.Column(variant="panel"):
            gapx = gr.Slider(label="Horizontal Gap",
                             value=128, minimum=16,
                             maximum=2048, step=16,)
            gapy = gr.Slider(label="Vertical Gap", minimum=16,
                             maximum=2048, step=16,

                             value=64)

        with gr.Column(variant="panel"):
            items_per_row = gr.Slider(label="Items per row", minimum=1,
                                      maximum=128, step=1,
                                      value=4)

        # folder = gr.Textbox(label="Folder",
        #                     value='/data/fractalcalendar')
            mask_blur = gr.Slider(label='Mask blur', minimum=0,
                                  maximum=256, step=1, value=4)
        with gr.Column():
            inpainting_fill = gr.Radio(label='Masked content', choices=[
                'fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index")

        return [folder, mask_blur, inpainting_fill, item_width, item_height, gapx, gapy, items_per_row]


# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, folder, mask_blur, inpainting_fill, item_width, item_height, gapx, gapy, items_per_row):

        p.mask_blur = mask_blur * 2
        p.inpainting_fill = inpainting_fill
        p.inpaint_full_res = False
        # read files fromt input directory to stitch

        stitchFiles = []
        for file in os.listdir(folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                LOGGER("Found image:", os.path.join(folder, file))
                stitchFiles.append(Image.open(os.path.join(folder, file)))
                continue
            else:
                continue

        basename = ""

        LOGGER("Preparing Image")
        border = gapx
        bordery = gapy
        columns = items_per_row
        stampWidth = p.width
        stampWidthHalve = stampWidth//2
        work_results = []
        target_w = border+(item_width+border)*columns
        target_h = bordery+(item_height+bordery) * \
            (math.ceil(len(stitchFiles)/columns))
        resultImage = Image.new("RGB", (target_w, target_h))
        resultMask = Image.new("L", (target_w, target_h), "white")
        resultLatentMask = Image.new("L", (target_w, target_h), "white")
        for index, image in enumerate(stitchFiles):
            i = index
            x1 = border+(item_width+border)*(i % columns)
            y1 = bordery + (item_height+bordery)*(math.ceil(i//columns))
            x2 = x1+item_width
            y2 = y1+item_height
            image.thumbnail((item_width, item_height),
                            Image.Resampling.LANCZOS)
            resultImage.paste(image, (x1, y1))
            # resultImage.paste(image, (x1, y1))

            draw = ImageDraw.Draw(resultMask)
            draw.rectangle((
                x1+mask_blur, y1+mask_blur*2, x2-mask_blur*2, y2-mask_blur*2
            ), fill="black")

            drawLatent = ImageDraw.Draw(resultLatentMask)
            drawLatent.rectangle((
                x1+mask_blur//2, y1+mask_blur//2, x2-mask_blur//2, y2-mask_blur//2
            ), fill="black")

        LOGGER("Image prepared", resultImage.size)
        xsteps = resultImage.width//stampWidthHalve
        ysteps = resultImage.height//stampWidthHalve

        state.job_count = ysteps*xsteps
        state.job_no
        current_seed = None
        for j in range(0, ysteps):
            if state.interrupted:
                # Interrupt button pressed in WebUI
                break
            for i in range(0, xsteps):
                if state.interrupted:
                    # Interrupt button pressed in WebUI
                    break
                state.job_no += 1
                state.job = "Calculating "+str(i)+"/"+str(j)
                # create current image to render
                img = Image.new("RGB", (p.width, p.height))
                # paste the incoming image into place
                img.paste(resultImage.crop(
                    (i*(stampWidthHalve), j*(stampWidthHalve), i*(stampWidthHalve)+stampWidth, j*(stampWidthHalve)+stampWidth)))

                #  ck the mask1 seems to indicate area to pertain
                mask = Image.new("L", (img.width, img.height), "white")

                mask.paste(resultMask.crop(
                    (i*(stampWidthHalve), j*(stampWidthHalve), i*(stampWidthHalve)+stampWidth, j*(stampWidthHalve)+stampWidth)))
                draw = ImageDraw.Draw(mask)
                if i > 0:
                    draw.rectangle((0,
                                    0,
                                    stampWidthHalve-mask_blur*2,
                                    stampWidth), fill='black')

                if j > 0:

                    draw.rectangle((0,
                                    0,
                                    stampWidth,
                                    stampWidthHalve-mask_blur*2), fill='black')

                latent_mask = Image.new("L", (img.width, img.height), "white")

                latent_mask.paste(resultLatentMask.crop(
                    (i*(stampWidthHalve), j*(stampWidthHalve), i*(stampWidthHalve)+stampWidth, j*(stampWidthHalve)+stampWidth)))

                draw = ImageDraw.Draw(latent_mask)
                if i > 0:
                    draw.rectangle((0, 0,
                                    stampWidthHalve-mask_blur//2,
                                    stampWidth), fill='black')
                if j > 0:
                    draw.rectangle((0,
                                    0,
                                    stampWidth,
                                    stampWidthHalve-mask_blur//2), fill='black')

                p.init_images = [img]
                p.image_mask = mask
                p.latent_mask = latent_mask

                if (current_seed != None):
                    p.seed = current_seed

                proc = process_images(p)
                if (current_seed == None):
                    current_seed = proc.seed

                work_results += proc.images

                resultImage.paste(
                    proc.images[0], (i*(stampWidthHalve), j*stampWidthHalve))

                # Use Last Image as input with the next one

        # rotate and flip each image in the processed images
        # use the save_images method from images.py to save
        # them.
        for i in range(len(proc.images)):
            images.save_image(proc.images[i], p.outpath_samples, basename,
                              proc.seed + i, proc.prompt, opts.samples_format, info=proc.info, p=p)

        work_results.append(resultImage)
        work_results.append(resultMask)
        work_results.append(resultLatentMask)
        return Processed(p, work_results)
