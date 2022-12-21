# ILC Automatic1111 Stable Diffusion Scripts

this repo contains scripts to be used with [Automatic1111 Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## What stands ILC for?

Its the akronym of the brand [I-Love-Chaos](https://www.etsy.com/shop/iLoveChaos) for popularizing [Chaos Theory](https://en.wikipedia.org/wiki/Chaos_theory). 

## Installation

copy the .py files into the ./scripts folder of the [Automatic1111 Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) installation

# Available Plugins

## ILC Simple Animator

This is a very simple plugin, it is used by c.k. in the youtube channel [ChaosTube](https://www.youtube.com/channel/UCej4aqqeusL5iUnKHjmKjLQ) for demonstrating what happens when the denoising parameter is changed. It has been extended to include the cfg scale parameter as well. Other than 
that the script is derived from [Animator Script](https://github.com/Animator-Anon/Animator) this script is included for reference wether anyone is interested in how it is done.

## ILC Image Stitcher

The script has been used to create the title image of the [Fractal Art Calendar 2023 - dawn of an era](https://www.etsy.com/listing/1343211412/fractal-art-calendar-2023). In principle this script
takes an input folder, arranges all found images in a rectangular manner and fills the gaps using outpainting.

Derived from the poor-mans-outpainting, it uses a consecutive way to create new images so that already created outpaints are incorporated into the next step.
