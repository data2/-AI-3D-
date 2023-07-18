from PIL import Image
import paddlehub as hub
module = hub.Module(name='disco_diffusion_ernievil_base')

result = module.generate_image(text_prompts="孤舟蓑笠翁，独钓寒江雪。", style='油画', width_height= [1280, 768], output_dir='孤舟蓑笠翁_油画', seed=1853109922)
display(Image.fromarray(result[0].load_uri_to_image_tensor().tensor))

result[0].chunks.save_gif('孤舟蓑笠翁.gif')

result = module.generate_image(text_prompts="孤舟蓑笠翁，独钓寒江雪。风格为水墨画。", output_dir='孤舟蓑笠翁_水墨画')
display(Image.fromarray(result[0].load_uri_to_image_tensor().tensor))
