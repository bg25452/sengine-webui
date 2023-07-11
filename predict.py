# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor,BaseModel, Input, Path, File
from replicate_call import WebUIApi,InstructPix2PixInterface,WebUIApiResult
import json
from collections import OrderedDict
from typing import Any,List
import sys
# sys.setdefaultencoding('utf-8')
# class Output(BaseModel):
#         images: List[File]
#         parameters: str
#         info: str
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.api = {'web':WebUIApi(),'ipp':InstructPix2PixInterface()}
    def predict(self,apitype:str,method_name:str,args:str,kwargs:str)->str:
        """Run a single prediction on the model"""
        if apitype not in ('web','ipp'):
            return f"ERROR! THe api type doesn't exist! {apitype}"
        
        method = getattr(self.api[apitype],method_name,None)
        if method is None:
            return f"ERROR! The method doesn't exist! {method_name}"       
        args = json.loads(args,object_pairs_hook=OrderedDict)
        args = [v for k,v in args.items()]
        kwargs = json.loads(kwargs)
        name =  kwargs.pop("name","public")
        if name == 'segment' or method_name == "seg_masks":
            ret = method(*args,**{'images':kwargs['images']})
            return ret
        elif name == 'segment_impaint':
            set_use_model = getattr(self.api['web'],'util_set_model')
            set_use_model('jinpai')
            kwargs['sam_inpainting'] = True
            ret = method(*args,**kwargs)
            ret = json.loads(ret)
            ret['parameters']['name'] = name
            return json.dumps(ret)
        else:
            if kwargs.get("sam_params",None) is not None and len(kwargs['sam_params']) > 0:
                name = "jinpai"
            elif name == "ohwx,home":
                name = "indoor"
            set_use_model = getattr(self.api['web'],'util_set_model')
            set_use_model(name)
            ret = method(*args,**kwargs)
            ret = json.loads(ret)
            ret['parameters']['name'] = name
            # realo = Output(images = [File(i) for i in ret['images']],info = ret['info'],parameters = ret['parameters'])
            return json.dumps(ret)

    # def predict(self,image: File = Input(description="Image to enlarge"))->str:
    #     import io
    #     import base64
    #     from PIL import Image
    #     def b64_img(image):
    #         if not isinstance(image, str):
    #             buffered = io.BytesIO()
    #             image.save(buffered, format="PNG")
    #             img_base64 = 'data:image/png;base64,' + str(base64.b64encode(buffered.getvalue()), 'utf-8')
    #             return img_base64
    #         else:
    #             return image

    #     def raw_b64_img(image):
    #         if not isinstance(image, str):
    #             # XXX controlnet only accepts RAW base64 without headers
    #             buffered = io.BytesIO()
    #             image.save(buffered, format="PNG")
    #             img_base64 = str(base64.b64encode(buffered.getvalue()), 'utf-8')
    #             return img_base64
    #         else:
    #             return image
    #     """local test method"""
    #     apitype = 'web'
    #     method_name = 'img2img'
    #     if apitype not in ('web','ipp'):
    #         return f"ERROR! THe api type doesn't exist! {apitype}"
        
    #     method = getattr(self.api[apitype],method_name,None)
    #     if method is None:
    #         return f"ERROR! The method doesn't exist! {method_name}"
    #     # only for test
    #     name =  "public"
    #     set_use_model = getattr(self.api['web'],'util_set_model')
    #     set_use_model(name)
    #     image = Image.open(image)
    #     # print("resize ??????")
    #     # image = image.resize((512,512))
    #     # test sam
    #     img_base64 = b64_img(image)
    #     sam_params = {'positive_points':[[100,100]], 'negative_points':[[200,200]]}
    #     kwargs = {"prompt": "pig", "controlnet_units": [],"images":[img_base64],
    #     "negative_prompt": "ugly, out of frame", "seed": -1, "styles": ["anime"], "cfg_scale": 13,"sam_params":sam_params}
    #     # test controlnet
    #     # img_base64 = raw_b64_img(image)
    #     # kwargs = {"prompt": "pig", "controlnet_units": [{"input_image":img_base64, "mask": img_base64, "module": "canny", \
    #     # "model": "control_sd15_canny [fef5e48e]", "weight": 1.0, "resize_mode": "Just Resize", "lowvram": False, "processor_res": 64,\
    #     #  "threshold_a": 64, "threshold_b": 64, "guidance": 1.0, "guidance_start": 0.0, "guidance_end": 1.0, "guessmode": True}],\
    #     #   "negative_prompt": "ugly, out of frame", "seed": -1, "styles": ["anime"], "cfg_scale": 13}
    #     # kwargs = json.loads(kwargs)
    #     ret = method(**kwargs)
    #     # realo = Output(images = [File(i) for i in ret['images']],info = ret['info'],parameters = ret['parameters'])
    #     return "OKKKKK"

