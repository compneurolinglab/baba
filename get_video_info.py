import ffmpeg

video_path = "/Users/fish/Library/CloudStorage/Dropbox-lamb/wangfish1999@gmail.com/baba/Data/22_1726473871.mp4"
probe = ffmpeg.probe(video_path, 
					v="error", 
					select_streams="v:0",  # 第一个视频流
					show_entries="stream=width,height,r_frame_rate")
					
video_info = probe["streams"][0]

# max_pixels = width * height
# fps = r_frame_rate

# {'width': 640,
#  'height': 368,
#  'r_frame_rate': '15/1', 
#  'disposition': {'default': 1,
#   'dub': 0,
#   'original': 0,
#   'comment': 0,
#   'lyrics': 0,
#   'karaoke': 0,
#   'forced': 0,
#   'hearing_impaired': 0,
#   'visual_impaired': 0,
#   'clean_effects': 0,
#   'attached_pic': 0,
#   'timed_thumbnails': 0,
#   'non_diegetic': 0,
#   'captions': 0,
#   'descriptions': 0,
#   'metadata': 0,
#   'dependent': 0,
#   'still_image': 0},
#  'tags': {'language': 'und',
#   'handler_name': 'VideoHandler',
#   'vendor_id': '[0][0][0][0]'}}
# 


probe = ffmpeg.probe(video_path, 
					v="error", 
					select_streams="a:0",  # 第一个音频流
					show_entries="stream=sample_rate,channels")
					
audio_info = probe["streams"][0]