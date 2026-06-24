"""
Retrieve comments for a list of YouTube videos using the YouTube Data API.

Reads a YouTube API key from a local file (default: .youtube_api_key, kept out
of version control) and writes one comments json file per video id to
output_dir. Video ids for which comments could not be retrieved are written
to a separate "broken ids" file so the run can be retried later.
"""
import argparse
import os
import json
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


def fetch_comments_for_video(youtube, video_id, max_results=100):
    video_comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            pageToken=next_page_token,
        )
        response = request.execute()
        for comment in response["items"]:
            video_comments.append({
                "comment_text_orig": comment["snippet"]["topLevelComment"]["snippet"]["textOriginal"],
                "comment_text_display": comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
                "comment_likes": comment["snippet"]["topLevelComment"]["snippet"]["likeCount"],
                "comment_replies": comment["snippet"]["totalReplyCount"],
                "comment_datetime": comment["snippet"]["topLevelComment"]["snippet"]["updatedAt"],
            })
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return video_comments


def main(args):
    api_key = open(args.api_key_file, "r").read().strip()
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=api_key)

    os.makedirs(args.output_dir, exist_ok=True)

    video_ids = [v.strip() for v in open(args.video_id_file, "r").readlines() if v.strip()]
    print(f"Number of videos: {len(video_ids)}")

    corrected_video_ids = []
    broken_video_ids = []
    for video_id in tqdm(video_ids):
        if os.path.exists(os.path.join(args.output_dir, f"{video_id}.json")):
            print(f"Skipping video {video_id}")
            continue
        try:
            video_comments = fetch_comments_for_video(youtube, video_id)
            print(f"{video_id} has {len(video_comments)} comments")
            corrected_video_ids.append(video_id)
            with open(os.path.join(args.output_dir, f"{video_id}.json"), "w") as f:
                json.dump(video_comments, f)
        except HttpError as e:
            print(f"Error retrieving comments for video {video_id}: {e}")
            broken_video_ids.append(video_id)
            continue

    print(f"Summary: {len(corrected_video_ids)} accessible videos, {len(broken_video_ids)} inaccessible")

    if args.corrected_video_id_file:
        with open(args.corrected_video_id_file, "w") as f:
            f.write("\n".join(corrected_video_ids))

    if args.broken_video_id_file:
        with open(args.broken_video_id_file, "w") as f:
            f.write("\n".join(broken_video_ids))


# Example usage (run from repo root):
#   python data_preprocessing/retrieve_youtube_comments.py --video_id_file data/video_ids.txt --output_dir data/youtube_comments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve YouTube comments for a list of video ids")
    parser.add_argument("--video_id_file", required=True, help="Text file with one YouTube video id per line")
    parser.add_argument("--output_dir", required=True, help="Directory to write one comments json file per video id")
    parser.add_argument("--api_key_file", default=".youtube_api_key", help="Path to a file containing a YouTube Data API key")
    parser.add_argument("--corrected_video_id_file", default=None, help="Optional path to write video ids whose comments were successfully retrieved")
    parser.add_argument("--broken_video_id_file", default=None, help="Optional path to write video ids whose comments could not be retrieved")
    args = parser.parse_args()
    main(args)
