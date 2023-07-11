import os
import sys
import time
import importlib
import signal
import re
import launch
launch.prepare_environment()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import import_hook, errors, extra_networks, ui_extra_networks_checkpoints
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

import torch

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork
import base64
import io
from PIL import Image
import numpy as np
import cv2
# if cmd_opts.server_name:
#     server_name = cmd_opts.server_name
# else:
#     server_name = "0.0.0.0" if cmd_opts.listen else None


def check_versions():
    if shared.cmd_opts.skip_version_check:
        return

    expected_torch_version = "1.13.1"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        errors.print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    expected_xformers_version = "0.0.16rc425"
    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            errors.print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())


def initialize():
    check_versions()

    extensions.list_extensions()
    localization.list_localizations(cmd_opts.localizations_dir)

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return
    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    modelloader.list_builtin_upscalers()
    modules.scripts.load_scripts()
    modelloader.load_upscalers()
    modules.sd_vae.refresh_vae_list()
    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    try:
        modules.sd_models.load_model()
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)

    shared.opts.data["sd_model_checkpoint"] = shared.sd_model.sd_checkpoint_info.title

    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)

    shared.reload_hypernetworks()

    ui_extra_networks.intialize()
    ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
    ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
    ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

    extra_networks.initialize()
    extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())

    if cmd_opts.tls_keyfile is not None and cmd_opts.tls_keyfile is not None:

        try:
            if not os.path.exists(cmd_opts.tls_keyfile):
                print("Invalid path to TLS keyfile given")
            if not os.path.exists(cmd_opts.tls_certfile):
                print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
        except TypeError:
            cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
            print("TLS setup invalid, running webui without TLS")
        else:
            print("Running with TLS")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def setup_cors(app):
    if cmd_opts.cors_allow_origins and cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])

#####################
'''
copy from webuiapi client https://github.com/mix1009/sdwebuiapi/blob/main/webuiapi/webuiapi.py
'''
import json
import requests
import io
import base64
from PIL import Image
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

class Upscaler(str, Enum):
    none = 'None'
    Lanczos = 'Lanczos'
    Nearest = 'Nearest'
    LDSR = 'LDSR'
    BSRGAN = 'BSRGAN'
    ESRGAN_4x = 'ESRGAN_4x'
    R_ESRGAN_General_4xV3 = 'R-ESRGAN General 4xV3'
    ScuNET_GAN = 'ScuNET GAN'
    ScuNET_PSNR = 'ScuNET PSNR'
    SwinIR_4x = 'SwinIR 4x'

class HiResUpscaler(str, Enum):
    none = 'None'
    Latent = 'Latent'
    LatentAntialiased = 'Latent (antialiased)'
    LatentBicubic = 'Latent (bicubic)'
    LatentBicubicAntialiased = 'Latent (bicubic antialiased)'
    LatentNearest = 'Latent (nearist)'
    LatentNearestExact = 'Latent (nearist-exact)'
    Lanczos = 'Lanczos'
    Nearest = 'Nearest'
    ESRGAN_4x = 'ESRGAN_4x'
    LDSR = 'LDSR'
    ScuNET_GAN = 'ScuNET GAN'
    ScuNET_PSNR = 'ScuNET PSNR'
    SwinIR_4x = 'SwinIR 4x'

@dataclass
class WebUIApiResult:
    images: list
    parameters: dict
    info: dict

    @property
    def image(self):
        return self.images[0]

def get_bounding_box(mask):
    segmentation = np.where(mask == 1)
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))
    return [x_min,x_max,y_min,y_max]

class ControlNetUnit:
    def __init__(self,
                input_image:str,  ### base64 encoded image
                mask: str = "",
                module: str = "canny",
                model: str = "control_sd15_canny",
                weight: float = 1.0,
                resize_mode: str = "Just Resize",
                lowvram: bool = False,
                processor_res: int = 64,
                threshold_a: float = 64,
                threshold_b: float = 64,
                guidance: float = 1.0,
                guidance_start: float = 0.0,
                guidance_end: float = 1.0,
                guessmode: bool = True):
        self.input_image = input_image
        self.mask = mask
        self.module = module
        self.model = model
        self.weight = weight
        self.resize_mode = resize_mode
        self.lowvram = lowvram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance = guidance
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.guessmode = guessmode

    def to_dict(self):
        return {
        "input_image": self.input_image if self.input_image else "",
        "mask": self.mask if self.mask else "",
        "module": self.module,
        "model": self.model,
        "weight": self.weight,  # controlnet 控制力 minimum=0.0, maximum=2.0, step=.05
        "resize_mode": self.resize_mode,
        "lowvram": self.lowvram,
        "processor_res": self.processor_res,
        "threshold_a": self.threshold_a,
        "threshold_b": self.threshold_b,
        "guidance": self.guidance,
        "guidance_start": self.guidance_start,
        "guidance_end": self.guidance_end,
        "guessmode": self.guessmode,
        }


def b64_img(image):
    if not isinstance(image, str):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = 'data:image/png;base64,' + str(base64.b64encode(buffered.getvalue()), 'utf-8')
        return img_base64
    else:
        return image

def raw_b64_img(image):
    if not isinstance(image, str):
        #  controlnet only accepts RAW base64 without headers
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = str(base64.b64encode(buffered.getvalue()), 'utf-8')
        return img_base64
    else:
        return image
    
def _to_api_result(response):
        json_func = getattr(response, "json", None)
        if callable(json_func):
            return response.json()
        else:
            return json.dumps(response)
        # print(dir(response))
        # if response.status_code != 200:
        #     raise RuntimeError(response.status_code, response.text)

        r = response
        images = []
        # print("type(r.images[0])",type(r.images[0]))
        if hasattr(r,'images'):
            images = [Image.open(io.BytesIO(base64.b64decode(i))) for i in r.images]
            # images = r.images
        elif hasattr(r,'image'):
            images = [Image.open(io.BytesIO(base64.b64decode(r.image)))]
            # images = [r.image]

        info = ''
        if hasattr(r,'info'):
            # try:
            #     info = json.loads(r.info)
            # except:
            #     info = r.info
            info = r.info
        elif hasattr(r,'html_info'):
            info = r.html_info
        # print(type(info))
        parameters = ''
        if hasattr(r,'parameters'):
            parameters = r.parameters
        # print(type(parameters))
        parameters = json.dumps(parameters)
        # return dict(images=images, parameters=parameters, info=info)
        return dict(images=images, parameters=parameters, info=info)  

class WebUIApi:
    def __init__(self,
                 sampler='Euler a',
                 steps=20):
        initialize()
        self.baseurl = '/sdapi/v1'
        from modules.api.api import Api
        self.api = Api()
        self.default_sampler = sampler
        self.default_steps = steps

    def txt2img(self,
                enable_hr=False,
                denoising_strength=0.0,
                firstphase_width=0,
                firstphase_height=0,
                hr_scale=2,
                hr_upscaler=HiResUpscaler.Latent,
                hr_second_pass_steps=0,
                hr_resize_x=0,
                hr_resize_y=0,
                prompt="shabi",
                styles=[],
                seed=-1,
                subseed=-1,
                subseed_strength=0.0,
                seed_resize_from_h=-1,
                seed_resize_from_w=-1,
                sampler_name=None,  # use this instead of sampler_index
                batch_size=1,
                n_iter=1,
                steps=None,
                cfg_scale=7.0,
                width=512,
                height=512,
                restore_faces=False,
                tiling=False,
                negative_prompt="",
                eta=0,
                s_churn=0,
                s_tmax=0,
                s_tmin=0,
                s_noise=1,
                override_settings={},
                override_settings_restore_afterwards=True,
                script_args=None,  # List of arguments for the script "script_name"
                script_name=None,
                controlnet_units=[],
                sampler_index=None, # deprecated: use sampler_name
                ):
        if sampler_index is None:
            sampler_index = self.default_sampler
        if sampler_name is None:
            sampler_name = self.default_sampler
        if steps is None:
            steps = self.default_steps
        if script_args is None:
            script_args = []
        payload = {
            "enable_hr": enable_hr,
            "hr_scale" : hr_scale,
            "hr_upscaler" : hr_upscaler,
            "hr_second_pass_steps" : hr_second_pass_steps,
            "hr_resize_x": hr_resize_x,
            "hr_resize_y": hr_resize_y,
            "denoising_strength": denoising_strength,
            "firstphase_width": firstphase_width,
            "firstphase_height": firstphase_height,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "seed_resize_from_h": seed_resize_from_h,
            "seed_resize_from_w": seed_resize_from_w,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "s_churn": s_churn,
            "s_tmax": s_tmax,
            "s_tmin": s_tmin,
            "s_noise": s_noise,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
            "sampler_name": sampler_name,
            "sampler_index": sampler_index,
            "script_name": script_name,
            "script_args": script_args
        }
        if controlnet_units and len(controlnet_units)>0:
            payload["controlnet_units"] = controlnet_units#[x.to_dict() for x in controlnet_units]
            return _to_api_result(self.api.call_api('/controlnet/txt2img', payload))
        else:
            response = self.api.call_api(f'{self.baseurl}/txt2img', payload)
            return _to_api_result(response)

    def seg_masks(self,images=[]):
        init_images = [b64_img(x) for x in images]
        # print("init_images",init_images)
        sam_params = {'input_image':init_images[0]} 
        masks_results = self.api.call_api('/sam/sam_masks', sam_params)
        # masks_results['mask_img'] = b64_img(masks_results['mask_img'])
            

        return json.dumps(masks_results)
        
        
    def img2img(self,
                images=[],  # list of PIL Image
                resize_mode=0,
                denoising_strength=0.75,
                image_cfg_scale=1.5,
                mask_image=None,  # PIL Image mask
                mask_blur=4,
                inpainting_fill=0,
                inpaint_full_res=True,
                inpaint_full_res_padding=0,
                inpainting_mask_invert=0,
                initial_noise_multiplier=1,
                prompt="",
                styles=[],
                seed=-1,
                subseed=-1,
                subseed_strength=0,
                seed_resize_from_h=-1,
                seed_resize_from_w=-1,
                sampler_name=None,  # use this instead of sampler_index
                batch_size=1,
                n_iter=1,
                steps=None,
                cfg_scale=7.0,
                width=512,
                height=512,
                restore_faces=False,
                tiling=False,
                negative_prompt="",
                eta=0,
                s_churn=0,
                s_tmax=0,
                s_tmin=0,
                s_noise=1,
                override_settings={},
                override_settings_restore_afterwards=True,
                script_args=None,  # List of arguments for the script "script_name"
                sampler_index=None,  # deprecated: use sampler_name
                include_init_images=False,
                script_name=None,
                controlnet_units=[],
                ### add by yan
                sam_params = {},
                sam_inpainting = False,
                ):
        if sampler_name is None:
            sampler_name = self.default_sampler
        if sampler_index is None:
            sampler_index = self.default_sampler
        if steps is None:
            steps = self.default_steps
        if script_args is None:
            script_args = []

        payload = {
            "init_images": [b64_img(x) for x in images],
            "resize_mode": resize_mode,
            "denoising_strength": denoising_strength,
            "mask_blur": mask_blur,
            "inpainting_fill": inpainting_fill,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpainting_mask_invert": inpainting_mask_invert,
            "initial_noise_multiplier": initial_noise_multiplier,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "seed_resize_from_h": seed_resize_from_h,
            "seed_resize_from_w": seed_resize_from_w,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "s_churn": s_churn,
            "s_tmax": s_tmax,
            "s_tmin": s_tmin,
            "s_noise": s_noise,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
            "sampler_name": sampler_name,
            "sampler_index": sampler_index,
            "include_init_images": include_init_images,
            "script_name": script_name,
            "script_args": script_args
        }
        if mask_image is not None:
            payload['mask'] = b64_img(mask_image)

        if len(sam_params) > 0:
            sam_params['input_image'] = payload['init_images'][0]
            mask_image = self.api.call_api('/sam/sam_predict', sam_params)
            payload['mask'] = b64_img(mask_image)

        if controlnet_units and len(controlnet_units)>0:
            payload["controlnet_units"] = controlnet_units#[x.to_dict() for x in controlnet_units]
            return _to_api_result(self.api.call_api('/controlnet/img2img', payload))
        elif sam_inpainting == True:
            response = self.api.call_api(f'{self.baseurl}/img2img', payload)
            whole_image = Image.open(io.BytesIO(base64.b64decode(response['images'][0])))
            whole_mask = Image.open(io.BytesIO(base64.b64decode(mask_image)))
            npimage = np.array(whole_image)
            npmask = np.array(whole_mask)
            bbox = get_bounding_box(npmask)
            crop_image = npimage[bbox[2]:bbox[3]+1,bbox[0]:bbox[1]+1,:]
            crop_mask = npmask[bbox[2]:bbox[3]+1,bbox[0]:bbox[1]+1]
            ones = np.ones((crop_image.shape[0],crop_image.shape[1],1),dtype='uint8') * 255
            crop_mask = np.stack([crop_mask,crop_mask,crop_mask,crop_mask],axis = -1)
            crop_image = np.concatenate([crop_image[:,:,:3], ones],axis=-1)
            crop_image = crop_image*crop_mask
            response['images'][0] = raw_b64_img(Image.fromarray(crop_image))
            return _to_api_result(response)
        else:
            response = self.api.call_api(f'{self.baseurl}/img2img', payload)
            return _to_api_result(response)

    def extra_single_image(self,
                           image,  # PIL Image
                           resize_mode=0,
                           show_extras_results=True,
                           gfpgan_visibility=0,
                           codeformer_visibility=0,
                           codeformer_weight=0,
                           upscaling_resize=2,
                           upscaling_resize_w=512,
                           upscaling_resize_h=512,
                           upscaling_crop=True,
                           upscaler_1="None",
                           upscaler_2="None",
                           extras_upscaler_2_visibility=0,
                           upscale_first=False,
                           ):
        payload = {
            "resize_mode": resize_mode,
            "show_extras_results": show_extras_results,
            "gfpgan_visibility": gfpgan_visibility,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "upscaling_resize": upscaling_resize,
            "upscaling_resize_w": upscaling_resize_w,
            "upscaling_resize_h": upscaling_resize_h,
            "upscaling_crop": upscaling_crop,
            "upscaler_1": upscaler_1,
            "upscaler_2": upscaler_2,
            "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
            "upscale_first": upscale_first,
            "image": b64_img(image),
        }

        response = self.api.call_api(f'{self.baseurl}/extra-single-image', payload)
        return _to_api_result(response)

    def extra_batch_images(self,
                           images,  # list of PIL images
                           name_list=None,  # list of image names
                           resize_mode=0,
                           show_extras_results=True,
                           gfpgan_visibility=0,
                           codeformer_visibility=0,
                           codeformer_weight=0,
                           upscaling_resize=2,
                           upscaling_resize_w=512,
                           upscaling_resize_h=512,
                           upscaling_crop=True,
                           upscaler_1="None",
                           upscaler_2="None",
                           extras_upscaler_2_visibility=0,
                           upscale_first=False,
                           ):
        if name_list is not None:
            if len(name_list) != len(images):
                raise RuntimeError('len(images) != len(name_list)')
        else:
            name_list = [f'image{i + 1:05}' for i in range(len(images))]
        images = [b64_img(x) for x in images]

        image_list = []
        for name, image in zip(name_list, images):
            image_list.append({
                "data": image,
                "name": name
            })

        payload = {
            "resize_mode": resize_mode,
            "show_extras_results": show_extras_results,
            "gfpgan_visibility": gfpgan_visibility,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "upscaling_resize": upscaling_resize,
            "upscaling_resize_w": upscaling_resize_w,
            "upscaling_resize_h": upscaling_resize_h,
            "upscaling_crop": upscaling_crop,
            "upscaler_1": upscaler_1,
            "upscaler_2": upscaler_2,
            "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
            "upscale_first": upscale_first,
            "imageList": image_list,
        }

        response = self.api.call_api(f'{self.baseurl}/extra-batch-images', payload)
        return _to_api_result(response)

    # XXX 500 error (2022/12/26)
    def png_info(self, image):
        payload = {
            "image": b64_img(image),
        }

        response = self.api.call_api(f'{self.baseurl}/png-info', payload)
        return _to_api_result(response)

    # XXX always returns empty info (2022/12/26)
    def interrogate(self, image):
        payload = {
            "image": b64_img(image),
        }

        response = self.api.call_api(f'{self.baseurl}/interrogate', payload)
        return _to_api_result(response)

    def get_options(self):
        response = self.api.call_api(f'{self.baseurl}/options')
        return response
    def set_options(self, options):
        response = self.api.call_api(f'{self.baseurl}/options', options)
        return response

    def get_progress(self):
        response = self.api.call_api(f'{self.baseurl}/progress')
        return response

    def get_cmd_flags(self):
        response = self.api.call_api(f'{self.baseurl}/cmd-flags')
        return response
    def get_samplers(self):        
        response = self.api.call_api(f'{self.baseurl}/samplers')
        return response
    def get_upscalers(self):        
        response = self.api.call_api(f'{self.baseurl}/upscalers')
        return response
    def get_sd_models(self):        
        response = self.api.call_api(f'{self.baseurl}/sd-models')
        return response
    def get_hypernetworks(self):        
        response = self.api.call_api(f'{self.baseurl}/hypernetworks')
        return response
    def get_face_restorers(self):        
        response = self.api.call_api(f'{self.baseurl}/face-restorers')
        return response
    def get_realesrgan_models(self):        
        response = self.api.call_api(f'{self.baseurl}/realesrgan-models')
        return response
    def get_prompt_styles(self):        
        response = self.api.call_api(f'{self.baseurl}/prompt-styles')
        return response
    def get_artist_categories(self):        
        response = self.api.call_api(f'{self.baseurl}/artist-categories')
        return response
    def get_artists(self):        
        response = self.api.call_api(f'{self.baseurl}/artists')
        return response
    def refresh_checkpoints(self):
        response = self.api.call_api(f'{self.baseurl}/refresh-checkpoints')
        return response

    def get_endpoint(self, endpoint, baseurl):
        if baseurl:
            return f'{self.baseurl}/{endpoint}'
        else:
            from urllib.parse import urlparse, urlunparse
            parsed_url = urlparse(self.baseurl)
            basehost = parsed_url.netloc
            parsed_url2 = (parsed_url[0], basehost, endpoint, '', '', '')
            return urlunparse(parsed_url2)


    def util_get_model_names(self):
        return sorted([x['title'] for x in self.get_sd_models()])
    def util_set_model(self, name, find_closest=True):
        if find_closest:
            name = name.lower()
        models = self.util_get_model_names()
        found_model = None
        if name in models:
            found_model = name
        elif find_closest:
            import difflib
            def str_simularity(a, b):
                return difflib.SequenceMatcher(None, a, b).ratio()
            max_sim = 0.0
            max_model = models[0]
            for model in models:
                sim = str_simularity(name, model)
                if sim >= max_sim:
                    max_sim = sim
                    max_model = model
            found_model = max_model
        if found_model:
            print(f'loading {found_model}')
            options = {}
            options['sd_model_checkpoint'] = found_model
            self.set_options(options)
            print(f'model changed to {found_model}')
        else:
            print('model not found')

    def util_get_current_model(self):
        return self.get_options()['sd_model_checkpoint']

    def util_wait_for_ready(self, check_interval=5.0):
        import time
        while True:
            result =  self.get_progress()
            progress = result['progress']
            job_count = result['state']['job_count']
            if progress == 0.0 and job_count == 0:
                break
            else:
                print(f'[WAIT]: progress = {progress:.4f}, job_count = {job_count}')
                time.sleep(check_interval)



## Interface for extensions

# https://github.com/mix1009/model-keyword
@dataclass
class ModelKeywordResult:
    keywords: list
    model: str
    oldhash: str
    match_source: str
    
class ModelKeywordInterface:
    def __init__(self, webuiapi):
        self.api = webuiapi
    def get_keywords(self):
        result = self.api.custom_get('model_keyword/get_keywords')
        keywords = result['keywords']
        model = result['model']
        oldhash = result['hash']
        match_source = result['match_source']
        return ModelKeywordResult(keywords, model, oldhash, match_source)

# https://github.com/Klace/stable-diffusion-webui-instruct-pix2pix
class InstructPix2PixInterface:
    def __init__(self):
        from modules.api.api import Api
        self.api = Api()
    def img2img(self, 
        images=[],
        prompt: str = '',
        negative_prompt: str = '',
        output_batches: int = 1,
        sampler: str = 'Euler a',
        steps: int = 20,
        seed: int = 0,
        randomize_seed: bool = True,
        text_cfg: float = 7.5,
        image_cfg: float = 1.5,
        randomize_cfg: bool = False,
        output_image_width: int = 512
        ):
        init_images = [b64_img(x) for x in images]
        payload = {
            "init_images": init_images,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_batches": output_batches,
            "sampler": sampler,
            "steps": steps,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "text_cfg": text_cfg,
            "image_cfg": image_cfg,
            "randomize_cfg": randomize_cfg,
            "output_image_width": output_image_width
        }
        return _to_api_result(self.api.call_api('/instruct-pix2pix/img2img', payload))


# https://github.com/Mikubill/sd-webui-controlnet
class ControlNetInterface:
    def __init__(self, webuiapi, show_deprecation_warning=True):
        from modules.api.api import Api
        self.api = Api()
        self.show_deprecation_warning = show_deprecation_warning
        
    def print_deprecation_warning(self):
        print('ControlNetInterface txt2img/img2img is deprecated. Please use normal txt2img/img2img with controlnet_units param')
        
    def txt2img(self,
        prompt: str = "",
        negative_prompt: str = "",
        controlnet_input_image: list = [],
        controlnet_mask: list = [],
        controlnet_module: str = "",
        controlnet_model: str = "",
        controlnet_weight: float = 0.5,
        controlnet_resize_mode: str = "Scale to Fit (Inner Fit)",
        controlnet_lowvram: bool = False,
        controlnet_processor_res: int = 512,
        controlnet_threshold_a: int = 64,
        controlnet_threshold_b: int = 64,
        controlnet_guidance: float = 1.0,
        enable_hr: bool = False, # hiresfix
        denoising_strength: float = 0.5,
        hr_scale: float = 1.5,
        hr_upscale: str = "Latent",
        guess_mode: bool = True,
        seed: int = -1,
        subseed: int = -1,
        subseed_strength: int = -1,
        sampler_index: str = "Euler a",
        batch_size: int = 1,
        n_iter: int = 1, # Iteration
        steps: int = 20,
        cfg_scale: float = 7,
        width: int = 512,
        height: int = 512,
        restore_faces: bool = False,
        override_settings: Dict[str, Any] = None, 
        override_settings_restore_afterwards: bool = True):
        
        if self.show_deprecation_warning:
            self.print_deprecation_warning()
        
        controlnet_input_image_b64 = [raw_b64_img(x) for x in controlnet_input_image]
        controlnet_mask_b64 = [raw_b64_img(x) for x in controlnet_mask]

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_input_image": controlnet_input_image_b64,
            "controlnet_mask": controlnet_mask_b64,
            "controlnet_module": controlnet_module,
            "controlnet_model": controlnet_model,
            "controlnet_weight": controlnet_weight,
            "controlnet_resize_mode": controlnet_resize_mode,
            "controlnet_lowvram": controlnet_lowvram,
            "controlnet_processor_res": controlnet_processor_res,
            "controlnet_threshold_a": controlnet_threshold_a,
            "controlnet_threshold_b": controlnet_threshold_b,
            "controlnet_guidance": controlnet_guidance,
            "enable_hr": enable_hr,
            "denoising_strength": denoising_strength,
            "hr_scale": hr_scale,
            "hr_upscale": hr_upscale,
            "guess_mode": guess_mode,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "sampler_index": sampler_index,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
        }
        return self.api.call_api('controlnet/txt2img', payload)
    
    def img2img(self,
        init_images: list = [],
        mask: str = None,
        mask_blur: int = 30,
        inpainting_fill: int = 0,
        inpaint_full_res: bool = True,
        inpaint_full_res_padding: int = 1,
        inpainting_mask_invert: int = 1,
        resize_mode: int = 0,
        denoising_strength: float = 0.7,
        prompt: str = "",
        negative_prompt: str = "",
        controlnet_input_image: list = [],
        controlnet_mask: list = [],
        controlnet_module: str = "",
        controlnet_model: str = "",
        controlnet_weight: float = 1.0,
        controlnet_resize_mode: str = "Scale to Fit (Inner Fit)",
        controlnet_lowvram: bool = False,
        controlnet_processor_res: int = 512,
        controlnet_threshold_a: int = 64,
        controlnet_threshold_b: int = 64,
        controlnet_guidance: float = 1.0,
        guess_mode: bool = True,
        seed: int = -1,
        subseed: int = -1,
        subseed_strength: int = -1,
        sampler_index: str = "",
        batch_size: int = 1,
        n_iter: int = 1, # Iteration
        steps: int = 20,
        cfg_scale: float = 7,
        width: int = 512,
        height: int = 512,
        restore_faces: bool = False,
        include_init_images: bool = True,
        override_settings: Dict[str, Any] = None, 
        override_settings_restore_afterwards: bool = True):
        
        if self.show_deprecation_warning:
            self.print_deprecation_warning()

        init_images_b64 = [raw_b64_img(x) for x in init_images]
        controlnet_input_image_b64 = [raw_b64_img(x) for x in controlnet_input_image]
        controlnet_mask_b64 = [raw_b64_img(x) for x in controlnet_mask]

        payload = {
            "init_images": init_images_b64,
            "mask": raw_b64_img(mask) if mask else None,
            "mask_blur": mask_blur,
            "inpainting_fill": inpainting_fill,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpainting_mask_invert": inpainting_mask_invert,
            "resize_mode": resize_mode,
            "denoising_strength": denoising_strength,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_input_image": controlnet_input_image_b64,
            "controlnet_mask": controlnet_mask_b64,
            "controlnet_module": controlnet_module,
            "controlnet_model": controlnet_model,
            "controlnet_weight": controlnet_weight,
            "controlnet_resize_mode": controlnet_resize_mode,
            "controlnet_lowvram": controlnet_lowvram,
            "controlnet_processor_res": controlnet_processor_res,
            "controlnet_threshold_a": controlnet_threshold_a,
            "controlnet_threshold_b": controlnet_threshold_b,
            "controlnet_guidance": controlnet_guidance,
            "guess_mode": guess_mode,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "sampler_index": sampler_index,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "include_init_images": include_init_images,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
        }
        return self.api.call_api('/controlnet/img2img', payload)
    
    def model_list(self):
        r = self.api.call_api('/controlnet/model_list')
        return r['model_list']


##test
# if __name__ == "__main__":
#     w = WebUIApi()
#     result1 = w.txt2img(prompt="cute squirrel",
#                     negative_prompt="ugly, out of frame",
#                     seed=1003,
#                     styles=["anime"],
#                     cfg_scale=7,
# #                      sampler_index='DDIM',
# #                      steps=30,
# #                      enable_hr=True,
# #                      hr_scale=2,
# #                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
# #                      hr_second_pass_steps=20,
# #                      hr_resize_x=1536,
# #                      hr_resize_y=1024,
# #                      denoising_strength=0.4,

#                     )
#     print(result1)

