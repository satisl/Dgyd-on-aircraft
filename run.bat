ffmpeg -i ""  -f rawvideo -pix_fmt yuv420p - 2>NUL | venv\Scripts\python.exe predict.py

pause