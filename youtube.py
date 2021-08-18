import pytube
from pytube.helpers import safe_filename


def get_yt_video(link, output_path):
    yt = pytube.YouTube(link)
    stream = yt.streams.filter(file_extension="mp4").first()
    stream.download(output_path = output_path)
    return safe_filename(yt.title)+".mp4"

if __name__ == '__main__':
    get_yt_video('https://www.youtube.com/watch?v=WvhYuDvH17I&ab_channel=masterryze', 'video_path')