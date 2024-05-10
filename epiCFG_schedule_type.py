import modules.scripts as scripts
import gradio as gr
import os
import math
from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
from math import floor

class ProcessedImagesWrapper:
    def __init__(self, images):
        self.images = images

    def js(self):
        flat_images = []
        for item in self.images:
            if isinstance(item, list):
                flat_images.extend([img for img in item if hasattr(img, 'js')])
            elif hasattr(item, 'js'):
                flat_images.append(item)
        return [image.js() for image in flat_images]

    @property
    def info(self):
        return "\n".join(str(image.info) for image in self.images if hasattr(image, 'info'))

    @property
    def comments(self):
        comments = [image.comments for image in self.images if hasattr(image, 'comments')]
        return "\n".join(comments) if comments else ""

class Script(scripts.Script):

    def title(self):
        return "epiCFG Schedule Type"

    def show(self, is_img2img):
        return not(is_img2img)

    def ui(self, is_img2img):
        schedule_options = [
            'Constant', 'Linear', 'Clamp-Linear (c=4.0)', 'Clamp-Linear (c=2.0)',
            'Clamp-Linear (c=1.0)', 'Inverse-Linear', 'PCS (s=0.01)', 'PCS (s=0.1)',
            'PCS (s=1.0)', 'PCS (s=2.0)', 'PCS (s=4.0)', 'Clamp-Cosine (c=4.0)',
            'Clamp-Cosine (c=2.0)', 'Clamp-Cosine (c=1.0)', 'Cosine', 'Sine',
            'V-Shape', 'A-Shape', 'Interval'
        ]
        schedule_multiselect_dropdown = gr.components.Dropdown(label="Schedule", choices=schedule_options, default="Inverse-Linear", multiselect=True)
        return [schedule_multiselect_dropdown]

    def run(self, p, schedules):
        strength = 1.0  # Fixed strength value
        processed_images = []
        if p.sampler_name in ('Euler a', 'Euler', 'LMS', 'DPM++ 2M', 'DPM fast', 'LMS Karras', 'DPM++ 2M Karras','DPM++ 2M SDE','DPM++ 3M SDE','Restart'):
            max_mul_count = p.steps * p.batch_size
            steps_per_mul = p.batch_size
        elif p.sampler_name in ('Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras', 'DPM++ SDE', 'DPM++ SDE Karras','UniPC'):
            max_mul_count = ((p.steps * 2) - 1) * p.batch_size
            steps_per_mul = 2 * p.batch_size
        elif p.sampler_name == 'DDIM':
            max_mul_count = fix_ddim_step_count(p.steps)
            steps_per_mul = 1
        elif p.sampler_name == 'UniPC':
            max_mul_count = fix_ddim_step_count(p.steps)
            steps_per_mul = 1
        elif p.sampler_name == 'PLMS':
            max_mul_count = fix_ddim_step_count(p.steps) + 1
            steps_per_mul = 1
        else:
            print('!!!warning: unsupported sampler ', p.sampler_name)
            return
        target_value = p.cfg_scale * (1 - strength)
        saved_obj = p.cfg_scale
        all_processed_images = []
        for schedule in schedules:
            print('\nepiCFG: ', schedule, end='\n')
            p.cfg_scale = Fake_float(p.cfg_scale, target_value, max_mul_count, steps_per_mul, p.steps, schedule)
            proc = process_images(p)
            processed_images = process_image_to_array(proc)
            all_processed_images.extend(processed_images)
            p.cfg_scale = saved_obj
        return ProcessedImagesWrapper(all_processed_images)

class Fake_float(float):
    def __new__(self, orig_value, target_value, max_mul_count, steps_per_mul, max_steps, schedule):
        return float.__new__(self, orig_value)

    def __init__(self, orig_value, target_value, max_mul_count, steps_per_mul, max_steps, schedule):
        float.__init__(orig_value)
        self.orig_value = orig_value
        self.target_value = target_value
        self.max_mul_count = max_mul_count
        self.current_mul = 0
        self.steps_per_mul = steps_per_mul
        self.current_step = 0
        self.max_step_count = (max_mul_count // steps_per_mul) + (max_mul_count % steps_per_mul > 0)
        self.max_steps = max_steps
        self.schedule = schedule

    def __mul__(self,other):
        return self.fake_mul(other)

    def __rmul__(self,other):
        return self.fake_mul(other)

    def fake_mul(self,other):
        if (self.max_step_count==1):
            fake_value= self.orig_value
        else:
            if self.schedule == 'Constant':
                fake_value = constant_schedule(self.current_step, self.max_steps, self.orig_value)
            elif self.schedule == 'Linear':
                fake_value = linear_schedule(self.current_step, self.max_steps, self.orig_value)
            elif self.schedule == 'Clamp-Linear (c=4.0)':
                fake_value = clamp_linear_schedule(self.current_step, self.max_steps, self.orig_value, 4.0)
            elif self.schedule == 'Clamp-Linear (c=2.0)':
                fake_value = clamp_linear_schedule(self.current_step, self.max_steps, self.orig_value, 2.0)
            elif self.schedule == 'Clamp-Linear (c=1.0)':
                fake_value = clamp_linear_schedule(self.current_step, self.max_steps, self.orig_value, 1.0)
            elif self.schedule == 'Inverse-Linear':
                fake_value = invlinear_schedule(self.current_step, self.max_steps, self.orig_value)
            elif self.schedule == 'PCS (s=0.01)':
                fake_value = powered_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 0.01)
            elif self.schedule == 'PCS (s=0.1)':
                fake_value = powered_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 0.1)
            elif self.schedule == 'PCS (s=1.0)':
                fake_value = powered_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 1.0)
            elif self.schedule == 'PCS (s=2.0)':
                fake_value = powered_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 2.0)
            elif self.schedule == 'PCS (s=4.0)':
                fake_value = powered_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 4.0)
            elif self.schedule == 'Clamp-Cosine (c=4.0)':
                fake_value = clamp_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 4.0)
            elif self.schedule == 'Clamp-Cosine (c=2.0)':
                fake_value = clamp_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 2.0)
            elif self.schedule == 'Clamp-Cosine (c=1.0)':
                fake_value = clamp_cosine_schedule(self.current_step, self.max_steps, self.orig_value, 1.0)
            elif self.schedule == 'Cosine':
                fake_value = cosine_schedule(self.current_step, self.max_steps, self.orig_value)
            elif self.schedule == 'Sine':
                fake_value = sine_schedule(self.current_step, self.max_steps, self.orig_value)
            elif self.schedule == 'V-Shape':
                fake_value = v_shape_schedule(self.current_step, self.max_steps, self.orig_value)
            elif self.schedule == 'A-Shape':
                fake_value = a_shape_schedule(self.current_step, self.max_steps, self.orig_value)
            elif self.schedule == 'Interval':
                fake_value = interval_schedule(self.current_step, self.max_steps, self.orig_value, 0.25, 5.42)
            else:
                print(f"Invalid CFG schedule: {self.schedule}")
                fake_value = self.orig_value
        self.current_mul = (self.current_mul+1) % self.max_mul_count
        self.current_step = (self.current_mul) // self.steps_per_mul
        return fake_value * other

def process_image_to_array(processed):
    if hasattr(processed, 'images') and isinstance(processed.images, list):
        return processed.images
    else:
        print("Processed object does not contain an iterable list of images.")
        return []

def fix_ddim_step_count(steps):
    valid_step = 999 / (1000 // steps)
    if valid_step == floor(valid_step): steps=int(valid_step)+1
    if ((1000 % steps)!=0): steps +=1
    return steps

def constant_schedule(step: int, max_steps: int, w0: float):
    return w0

def linear_schedule(step: int, max_steps: int, w0: float):
    return w0 * 2 * (1 - step / max_steps)

def clamp_linear_schedule(step: int, max_steps: int, w0: float, c: float):
    return max(c, linear_schedule(step, max_steps, w0))

def clamp_cosine_schedule(step: int, max_steps: int, w0: float, c: float):
    return max(c, cosine_schedule(step, max_steps, w0))

def invlinear_schedule(step: int, max_steps: int, w0: float):
    return w0 * 2 * (step / max_steps)

def powered_cosine_schedule(step: int, max_steps: int, w0: float, s: float):
    return w0 * ((1 - math.cos(math.pi * ((max_steps - step) / max_steps)**s))/2.0)

def cosine_schedule(step: int, max_steps: int, w0: float):
    return w0 * (1 + math.cos(math.pi * step / max_steps))

def sine_schedule(step: int, max_steps: int, w0: float):
    return w0 * (math.sin((math.pi * step / max_steps) - (math.pi / 2)) + 1) 

def v_shape_schedule(step: int, max_steps: int, w0: float):
    if step < max_steps / 2:
        return invlinear_schedule(step, max_steps, w0)
    return linear_schedule(step, max_steps, w0)

def a_shape_schedule(step: int, max_steps: int, w0: float):
    if step < max_steps / 2:
        return linear_schedule(step, max_steps, w0)
    return invlinear_schedule(step, max_steps, w0)

def interval_schedule(step: int, max_steps: int, w0: float, low: float, high: float):
    if low <= step <= high:
        return w0
    return 1.0
