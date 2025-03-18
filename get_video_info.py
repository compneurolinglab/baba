import ffmpeg

# get video info
video_path = "/Users/fish/Library/CloudStorage/Dropbox-lamb/wangfish1999@gmail.com/baba/Data/22_1726473871.mp4"
probe = ffmpeg.probe(video_path, 
					v="error", 
					select_streams="v:0",  # first video stream
					show_entries="stream=width,height,r_frame_rate")
					
video_info = probe["streams"][0]

# get audio info
probe = ffmpeg.probe(video_path, 
					v="error", 
					select_streams="a:0",  # first audio stream
					show_entries="stream=sample_rate,channels")
					
audio_info = probe["streams"][0]
