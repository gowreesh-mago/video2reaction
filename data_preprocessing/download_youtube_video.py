"""
Download videos from YouTube into output_dir.

The input is a text file with one YouTube URL (or video id) per line. Each
video gets its own folder named after the video id, containing the video file
(raw.mp4), the info json file (info.json), and optionally the comments json
file (comments.json).
"""
import argparse
import json
import time

from tqdm import tqdm
import yt_dlp as ytdl
import os
import random
from datetime import datetime

RAW_VIDEO_FILE_NAME = "raw.mp4"
TRIMMED_RAW_VIDEO_FILE_NAME = "raw_trimmed.mp4"
INFO_FILE_NAME = "info.json"
COMMENT_FILE_NAME = "comments.json"
VIDEO_DOWNLOADING_STATUS_FILE_NAME = "video_downloading_status.ndjson"


def check_video_downloaded(video_id, output_base_dir):
    if os.path.exists(os.path.join(output_base_dir, video_id, RAW_VIDEO_FILE_NAME)):
        return True
    if os.path.exists(os.path.join(output_base_dir, video_id, TRIMMED_RAW_VIDEO_FILE_NAME)):
        return True
    return False


def download_youtube_video(video_url,
                           output_base_dir,
                           video_info,
                           force_redownload=False,
                           cookie_file=None,
                           ):

    if not video_url.startswith("https"):
        video_url = f"https://www.youtube.com/watch?v={video_url}"
    if video_url.startswith("https://www.youtube.com/watch?v="):
        video_id = video_url.split("v=")[1]
    else:
        return {
            "youtube_url": video_url,
            'error': True,
            'error_message': "Invalid youtube url",
        }

    if check_video_downloaded(video_id, output_base_dir) and not force_redownload:
        print(f"Video already exists for {video_url}. Skip downloading")
        return None
    # Delete the videos that might have been downloaded in different format and resolution
    for video_file in os.listdir(os.path.join(output_base_dir, video_id)):
        if video_file.startswith("raw") and not video_file.endswith(".mp4"):
            os.remove(os.path.join(output_base_dir, video_id, video_file))
    if video_info["ext"] != "mp4":
        print(f"Skip downloading video {video_url} because it is not in mp4 format")
        return {
            "youtube_url": video_url,
            'error': True,
            'error_message': "Video is not in mp4 format",
        }

    try:
        ydl_opts = {
            'outtmpl': os.path.join(output_base_dir, '%(id)s', RAW_VIDEO_FILE_NAME),
            'format': f'bestquality[height<=360]+bestaudio/best',
            'noplaylist': True,
            'norepeats': True,
            'cookiefile': cookie_file,
            'sleep_interval_requests': 1,
            'sleep_interval': 5,
            'max_sleep_interval': 10,
        }
        with ytdl.YoutubeDL(ydl_opts) as ydl:
            print(f"Download video for {video_url}")
            ydl.download([video_url])
            return {
                "youtube_url": video_url,
                "youtube_video_id": video_id,
                "video_path": os.path.join(output_base_dir, video_id),
                'error': False,
            }

    except Exception as e:
        print(f"error: {e}")
        return {
            "youtube_url": video_url,
            'error': True,
            'error_message': str(e),
        }


def read_video_urls(input_file):
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def get_video_info(video_url, output_base_dir,
                   get_comments=False,
                   max_comments=10,
                   reprocess=False, cookie_file=None):
    if not video_url.startswith("https"):
        video_id = video_url
        video_url = f"https://www.youtube.com/watch?v={video_id}"
    if video_url.startswith("https://www.youtube.com/watch?v="):
        video_id = video_url.split("v=")[1]
    else:
        raise Exception("Invalid youtube url")
    extract = True
    available = True
    os.makedirs(os.path.join(output_base_dir, video_id), exist_ok=True)
    if os.path.exists(os.path.join(output_base_dir, video_id, INFO_FILE_NAME)) and not reprocess:
        # first check whether the video is available
        try:
            video_info = json.load(open(os.path.join(output_base_dir, video_id, INFO_FILE_NAME), "r"))
        except Exception as e:
            video_info = {"error": True, "error_message": str(e)}
        if video_info.get('error', False):
            if "Private video" in video_info["error_message"] or "mp4" in video_info["error_message"]:
                extract = False
                available = False
        else:
            if get_comments and os.path.exists(os.path.join(output_base_dir, video_id, COMMENT_FILE_NAME)):
                print(f"Video info and comments already exists for {video_url}. Skip downloading")
                extract = False
            if get_comments and not os.path.exists(os.path.join(output_base_dir, video_id, COMMENT_FILE_NAME)):
                print(f"Video info exists but comments do not exist for {video_url}.")
                extract = True
            if not get_comments:
                print(f"Video info already exists for {video_url}. Skip downloading")
                extract = False
    if not extract:
        return video_info, extract, available

    ydl_opts = {
            'format': f'bestquality[height<=360]+bestaudio/best',
            'noplaylist': True,
            'norepeats': True,
            'getcomments': get_comments,
            "extractor_args": {
            "youtube": {
                "comment_sort": "top",
                "max_comments": [
                    str(max_comments),
                    str(max_comments),
                    str(0),
                    str(0),
                ],
            }
        },
            'writecomments': get_comments,
            'cookiefile': cookie_file,
            'sleep_interval_requests': 1,
            'sleep_interval': 5,
            'max_sleep_interval': 10,
        }
    with ytdl.YoutubeDL(ydl_opts) as ytd:
        try:
            info = ytd.extract_info(video_url, download=False)
        except Exception as e:
            print(f"Error: {e}")
            video_info = {
                "youtube_url": video_url,
                'error': True,
                'error_message': str(e),
            }
            extracted = False
            available = False
            with open(os.path.join(output_base_dir, video_id, INFO_FILE_NAME), 'w') as f:
                json.dump(video_info, f)
            return video_info, extracted, available
        video_output_dir = os.path.join(output_base_dir, video_id)

        # Separate the comments from the info
        if get_comments:
            comments = info.get("comments", [])
            with open(os.path.join(video_output_dir, COMMENT_FILE_NAME), 'w') as f:
                json.dump(comments, f)
            info.pop("comments")
        with open(os.path.join(output_base_dir, info["id"], INFO_FILE_NAME), 'w') as f:
            json.dump(info, f)
    return info, extract, available


def main():
    # Example usage (run from repo root):
    #   python data_preprocessing/download_youtube_video.py --input_fpath data/video_urls.txt --output_dir data/raw_video --download_video
    #   python data_preprocessing/download_youtube_video.py --input_fpath data/video_urls.txt --output_dir data/raw_video --write_comments --cookie_file ~/.config/cookies.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fpath", help="Input text file containing URLs (one per line)", required=True)
    parser.add_argument("--output_dir", help="Output directory to save the downloaded videos", required=True)
    parser.add_argument("--force_reextract", help="Force re-extraction of the video info", action="store_true")
    parser.add_argument("--force_redownload", help="Force redownload of the videos", action="store_true")
    parser.add_argument("--write_comments", help="Comment download", action="store_true")
    parser.add_argument("--download_video", help="Download video", action="store_true")
    parser.add_argument("--min_comments", help="Minimum number of comments to download video", default=50, type=int)
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument("--cookie_file", help="Cookie file for youtube", default=None)
    parser.add_argument("--max_videos", help="Maximum number of videos to download", default=None, type=int)
    args = parser.parse_args()
    input_file_path = args.input_fpath

    if not os.path.isfile(input_file_path):
        raise Exception(f"Error: The file '{input_file_path}' does not exist")

    video_urls = read_video_urls(input_file_path)
    print(f"{len(video_urls)} can be downloaded")
    if args.max_videos:
        print(f"Downloading {args.max_videos} videos only")

    extracted_videos = 0
    downloaded_videos = 0
    unavailable_videos = 0
    for video_url in tqdm(video_urls):
        video_info, extracted, available = get_video_info(video_url,
                                    output_base_dir=args.output_dir,
                                    get_comments=args.write_comments,
                                    max_comments=500,
                                    reprocess=args.force_reextract,
                                    cookie_file=args.cookie_file)

        if video_info.get('error', False):
            download_status = video_info
            print(f"Something wrong with this video so skip downloading")
        else:
            if args.download_video:
                # Check comment threshold
                if video_info["comment_count"] is None or video_info["comment_count"] <= args.min_comments:
                    print(f"Skip downloading video {video_url} because it has less than {args.min_comments} comments")
                    download_status = None
                else:
                    download_status = download_youtube_video(video_url,
                                                 output_base_dir=args.output_dir,
                                                 video_info=video_info,
                                                 force_redownload=args.force_redownload,
                                                 cookie_file=args.cookie_file)
            else:
                download_status = None

        if extracted:
            extracted_videos += 1
        if download_status != None:
            downloaded_videos += 1
        if not available:
            unavailable_videos += 1

        # Add random sleep to avoid getting rate limited
        if extracted:
            time.sleep(random.randint(10, 20))
        if download_status != None and not download_status.get('error', False):
            time.sleep(random.randint(20, 60))

        # Every 100 downloaded videos, sleep more
        if download_status != None and not download_status.get('error', False) and downloaded_videos > 0 and downloaded_videos % 100 == 0:
            print(f"Downloaded {downloaded_videos} videos. Sleep for a longer period")
            time.sleep(random.randint(100, 300))

        if download_status != None and downloaded_videos > 0 and downloaded_videos % 500 == 0:
            print(f"Downloaded {downloaded_videos} videos. Sleep for a longer period")
            time.sleep(random.randint(300, 600))

        if download_status != None:
            with open(os.path.join(args.output_dir, VIDEO_DOWNLOADING_STATUS_FILE_NAME), 'a') as f:
                json.dump(download_status, f)
                f.write('\n')

        if args.max_videos and downloaded_videos >= int(args.max_videos):
            print(f"Reached maximum number of videos to download so stopped")
            timenow = datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Summary ({timenow}): {extracted_videos} videos extracted; {downloaded_videos} videos downloaded; {unavailable_videos} videos unavailable")
            break

        if args.debug:
            print(f"Debug mode")
            break


if __name__ == '__main__':
    main()
