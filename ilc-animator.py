#
# ILC Animation Script v0.1
#
# Forked from https://github.com/Animator-Anon/Animator
#
# Must have ffmpeg installed in path.
#
# The main idea for the script was to smoothly only animate one parameter - the denoising parameter.
#
#
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import os
import time
from scripts.ilc_mcommon import getILCBase64ImageSingleton, makePilILCMandelbrotLogo, ILC_LOGGER
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, images, sd_models
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state
import random
import subprocess
import numpy as np
import json
import cv2
import math

ILC_PLUGIN_NAME = "ILC-Simple-Animator"
ILC_PLUGIN_DESCRIPTION = """
Meant to play around with plugin development. 

It is used by Christian K. for animating just denoise and cfg_scale parameters. Christian is hosting the i-love-chaos brand, mainly a technology adoring place with special interest to deterministic chaos. 

These plugins are here for reference wether anyone is interested in seeing how something actually was done.
"""
ILC_PLUGIN_VERSION = 1.0
ILC_COMPLETE_NAME = ILC_PLUGIN_NAME+'@'+str(ILC_PLUGIN_VERSION)

LOGGER = ILC_LOGGER(ILC_PLUGIN_NAME, ILC_PLUGIN_VERSION)


def easeInOutExpo(x: float) -> float:
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return math.pow(2, 20 * x - 10) / 2
    if x >= 0.5:
        return (2 - math.pow(2, -20 * x + 10)) / 2
    return 0


def easeInOutSine(x):
    return -(math.cos(math.pi * x) - 1) / 2


def easeInOutCubic(x):
    if x < 0.5:
        return 4*x*x*x
    if x >= 0.5:
        return 1-math.pow(-2*x+2, 3)/2


def lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def make_gif(filepath, filename, fps, create_vid, create_bat):
    # Create filenames
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.gif"

    LOGGER("GIF", filename)

    # Build cmd for bat output, local file refs only
    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        out_filename
    ]
    # create bat file
    if create_bat:
        LOGGER(
            'Creating .bat file, use it when calculation is interrupted, or to preview output')
        with open(os.path.join(filepath, filename+'-'+"makegif.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd), "\r\n", "pause"])
    # Fix paths for normal output
    cmd[5] = os.path.join(filepath, in_filename)
    cmd[6] = os.path.join(filepath, out_filename)
    # create output if requested
    if create_vid:
        LOGGER('Creating GIF video file - yes, a gif is a video file as well now',
               filepath, out_filename)
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#        stdout, stderr = process.communicate()
#        if process.returncode != 0:
#            LOGGER(stderr)
#            raise RuntimeError(stderr)

def make_webm(filepath, filename, fps, create_vid, create_bat):
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.webm"

    LOGGER("WEBM", filename)
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-crf', str(50),
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        LOGGER(
            'Creating .bat file, use it when calculation is interrupted, or to preview output')
        with open(os.path.join(filepath, filename+'-'+"makewebm.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd), "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[10] = os.path.join(filepath, out_filename)

    if create_vid:
        LOGGER('Creating WEBM video file', filepath, out_filename)
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#        stdout, stderr = process.communicate()
#        if process.returncode != 0:
#            LOGGER(stderr)
#            raise RuntimeError(stderr)

def make_mp4(filepath, filename, fps, create_vid, create_bat):
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.mp4"
    LOGGER("MP4", filename)

    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        LOGGER(
            'Creating .bat file, use it when calculation is interrupted, or to preview output')
        with open(os.path.join(filepath, filename+'-' + "makemp4.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd), "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[16] = os.path.join(filepath, out_filename)

    if create_vid:
        LOGGER('Creating MP4 video file', filepath, out_filename)
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#        stdout, stderr = process.communicate()
#        if process.returncode != 0:
#            LOGGER(stderr)
#            raise RuntimeError(stderr)

class Script(scripts.Script):
    def title(self):
        return ILC_COMPLETE_NAME

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):

        with gr.Blocks() as demo22:
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

"""
                                )
                    # gr.Image(makePilILCMandelbrotLogo())
                with gr.Column():
                    with gr.Tab("Edit"):
                        with gr.Column(visible=is_img2img):
                            denoise_sliderstart = gr.Slider(
                                label="denoise_start", minimum=0.0, maximum=1.0, value=0.0)
                            denoise_sliderend = gr.Slider(
                                label="denoise_end", minimum=0.0, maximum=1.0, value=1.0)
                        with gr.Column():
                            cfgscalesliderstart = gr.Slider(
                                label="cfg scale start", minimum=0.0, maximum=30.0, value=5)
                            cfgscale_sliderend = gr.Slider(
                                label="cfg scale end", minimum=0.0, maximum=30.0, value=7)

                    # end ck

                        with gr.Row():
                            vid_gif = gr.Checkbox(label="GIF", value=False)
                            vid_mp4 = gr.Checkbox(label="MP4", value=False)
                            vid_webm = gr.Checkbox(label="WEBM", value=True)

                        with gr.Column():
                            totaltime = gr.Slider(
                                label="Total Animation Length (s)", minimum=0.0, maximum=600.0, value=30)
                            fps = gr.Slider(
                                label="Framerate", minimum=0.0, maximum=120, value=15, step=1)

                    with gr.Tab("Info/Help"):
                        gr.Markdown("""
##### ILC Simple Animator
    
"""+ILC_PLUGIN_DESCRIPTION+"""

The interpolation is eased by SineInOut.

###### Reference:
| Parameter  | Description  | 
|---|---|
|  denoise_start/end | the denoise animation values |  
| cfg scale start/end  |  The cfg scale animation values |   
| animation length|  The length of the animation in seconds |  
| fps |  The frames to be generated per second |  

 
            """
                                    )

                        i1 = gr.Markdown(
                            """
                                            ##### Save videos

                                            'ffmpeg' is required to be found in $PATH be sure to have it in place before starting webui:

                                            Windows PowerShell:

                                                $env:Path +=";[PATH TO YOUR FFMPEG BIN]"

                                            
                                                            """
                        )

        return [totaltime, fps, vid_gif, vid_mp4, vid_webm,   denoise_sliderstart, denoise_sliderend, cfgscalesliderstart, cfgscale_sliderend]

    def run(self, p,     totaltimeIn, fpsIn, vid_gif, vid_mp4, vid_webm,  denoise_start, denoise_end, cfg_scale_start, cfg_scale_end):

        initial_seed_save = p.seed

        all_images_collection = []
        all_images = []

        # for step in range(0,p.n_iter):
        # Fix variable types, i.e. text boxes giving strings.
        p.seed = initial_seed_save
        totaltime = float(totaltimeIn)
        fps = float(fpsIn)
        cfg_scale_end = float(cfg_scale_end)
        cfg_scale_start = float(cfg_scale_start)
        denoise_start = float(denoise_start)
        denoise_end = float(denoise_end)
        apply_colour_corrections = False

        outfilename = time.strftime('%Y%m%d%H%M%S')
        outpath = os.path.join(p.outpath_samples, outfilename)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        processing.fix_seed(p)
        batch_count = p.n_iter

        # Save extra parameters for the UI
        p.extra_generation_params = {
            "Create GIF": vid_gif,
            "Create MP4": vid_mp4,
            "Create WEBM": vid_webm,
            "Total Time (s)": totaltime,
            "FPS": fps,
            "Denoise Start": denoise_start,
            "Denoise End": denoise_end,
            "CFG Scale Value Start": cfg_scale_start,
            "CFG Scale Value End": cfg_scale_end,
            "End": denoise_end,
        }

        # save settings, just dump out the extra_generation dict
        settings_filename = os.path.join(
            outpath, f"{str(outfilename)}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            json.dump(dict(p.extra_generation_params),
                      f, ensure_ascii=False, indent=4)

        # This doesn't work, still some information missing if you don't drop an image into the img2img page.
        # if p.init_images[0] is None:
        #    a = np.random.rand(p.width, p.height, 3) * 255
        #    p.init_images.append(Image.fromarray(a.astype('uint8')).convert('RGB'))

        # p.n_iter = 1

        # output_images, info = None, None
        initial_seed = None
        initial_info = None

        # Make bat files before we start rendering video, so we could run them manually to preview output.
        for i in range(0, p.n_iter):
            make_gif(outpath, str(i+1)+'-'+outfilename, fps, False, True)
            make_mp4(outpath,  str(i+1)+'-'+outfilename, fps, False, True)
            make_webm(outpath, str(i+1)+'-' + outfilename, fps, False, True)

        frame_count = int(fps * totaltime)
        p.batch_size = 1
        state.job_count = frame_count * batch_count

        # initial_color_corrections = [
        #     processing.setup_color_correction(p.init_images[0])]

        # Iterate through range of frames
        for frame_no in range(frame_count):

            if state.interrupted:
                # Interrupt button pressed in WebUI
                break

            normalized_frame = (frame_no)/(frame_count-1)
            eased_frame = easeInOutSine(normalized_frame)
            # ck full interpolation 0...1 of denoising strength hard coded
            p.denoising_strength = lerp(
                denoise_start, denoise_end, eased_frame)
            p.cfg_scale = lerp(
                cfg_scale_start, cfg_scale_end, eased_frame)
            LOGGER("Denoise:", p.denoising_strength)
            LOGGER("cfg_scale", p.cfg_scale)
            LOGGER("seed", p.seed)
            LOGGER("frame", frame_no+1, "/", frame_count)
            state.job = f"Iteration {frame_no + 1}/{frame_count}"

            # Process current frame
            processed = processing.process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info
            index = 0
            for imga in processed.images:
                index = index+1
                imga.save(os.path.join(
                    outpath, f"{index}-{outfilename}_{frame_no:05}.png"))
                if (len(all_images_collection) < index):
                    all_images_collection.append([])
                if (frame_no % int(fps) == 0 or frame_no == frame_count-1):
                    # all_images.append(imga)
                    all_images_collection[index-1].append(imga)

        for i in range(0, p.n_iter):
            try:     # If not interrupted, make requested movies. Otherise the bat files exist.
                make_gif(outpath, str(i+1)+'-' + outfilename, fps, vid_gif &
                         (not state.interrupted), False)
                make_mp4(outpath, str(i+1)+'-' + outfilename, fps, vid_mp4 &
                         (not state.interrupted), False)
                make_webm(outpath, str(i+1)+'-' + outfilename, fps, vid_webm &
                          (not state.interrupted), False)
            except Exception as e:
                LOGGER("Error happened", e)
                LOGGER(
                    "Presumably a missing ffmpeg installation is the cause, make sure it is in the search $PATH")
        dickies = []
        for images in all_images_collection:
            for image in images:
                dickies.append(image)

        processed = Processed(p, dickies, initial_seed, initial_info)

        return processed
