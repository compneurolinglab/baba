import ffmpeg

video_path = "22_1726473871.mp4"
probe = ffmpeg.probe(video_path,
		     v="error", 
		     select_streams="v:0", 
		     show_entries="stream=width,height,r_frame_rate")					
video_info = probe["streams"][0]

probe = ffmpeg.probe(video_path,
		     v="error", 
		     select_streams="a:0",  
		     show_entries="stream=sample_rate,channels")
audio_info = probe["streams"][0]
