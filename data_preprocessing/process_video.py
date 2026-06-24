"""
Given a raw mp4 video, this script will:
1. Extract key frames using scene detection (drop the last few frames if they contain channel ads)
2. Pass key frames to a pretrained visual encoder and save visual embeddings
3. Extract audio waveform using ffmpeg
4. Pass audio waveform to an audio encoder (CLAP/HuBERT) to get audio embeddings
"""

import os
from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector, HistogramDetector
from scenedetect.scene_manager import save_images
from scenedetect import detect, SceneList, FrameTimecode
from tqdm import tqdm
import pandas as pd
from data_preprocessing.vision_encoder_zoo import VISION_ENCODER_ZOO
from data_preprocessing.audio_encoder_zoo import AUDIO_ENCODER_ZOO
from PIL import Image
import numpy as np
import torch
import torchaudio
from collections import defaultdict

torch.set_grad_enabled(False)


def trim_video_outro(video_path, video_duration_s: int, remove_untrimmed=False):
    """
    Source: CondensedMovies
    """
    if video_duration_s is None:
        print(f"No duration provided for video {video_path} so just use raw video path")
        return video_path
    trimmed_fp = video_path.replace("raw.mp4", "raw_trimmed.mp4")
    if os.path.exists(trimmed_fp):
        return trimmed_fp
    try:
        print(f"Trimming video {video_path}")
        cmd = 'ffmpeg -y -ss 0 -i {} -t {} -c copy {}'.format(video_path, video_duration_s, trimmed_fp)
        os.system(cmd)
    except Exception as e:
        print(f"Error trimming video {video_path}: {e}")
        return video_path
    if remove_untrimmed:
        os.remove(video_path)
    return trimmed_fp


def extract_keyframes(video_path, output_dir, min_num_frames=16):
    os.makedirs(output_dir, exist_ok=True)

    scene_list = detect(video_path, AdaptiveDetector(), start_in_scene=True)

    if len(scene_list) < min_num_frames:
        print(f"There are only {len(scene_list)} detected scenes in the video {video_path} so we will split into 10-second scenes instead")
        framerate = scene_list[0][0].get_framerate()
        start_frame = scene_list[0][0].get_frames()
        end_frame = scene_list[-1][1].get_frames()
        sampling_rate = min(int(framerate * 10), (end_frame - start_frame) // min_num_frames + 1)
        selected_frames = range(start_frame, end_frame, sampling_rate)
        scene_list = []
        for i in range(len(selected_frames) - 1):
            start = FrameTimecode(int(selected_frames[i]), fps=framerate)
            end = FrameTimecode(int(selected_frames[i + 1]), fps=framerate)
            scene_list.append((start, end))

    save_images(scene_list=scene_list,
                video=open_video(video_path),
                num_images=1, output_dir=output_dir,
                image_name_template='$SCENE_NUMBER')

    key_frame_list = []
    for scene_number, scene in enumerate(scene_list):
        key_frame_list.append({
            'scene_number': scene_number,
            'start_time': scene[0].get_timecode(),
            'end_time': scene[1].get_timecode(),
            'start_frame': scene[0].get_frames(),
            'end_frame': scene[1].get_frames(),
        })
    key_frame_df = pd.DataFrame(key_frame_list)
    key_frame_df.to_csv(f"{output_dir}/index.csv", index=False)
    return key_frame_df


def load_key_frames(key_frame_dir) -> np.ndarray:
    try:
        key_frame_list = pd.read_csv(f"{key_frame_dir}/index.csv")
    except pd.errors.EmptyDataError:
        print(f"Empty key frame index file {key_frame_dir}/index.csv")
        return False

    scene_list = key_frame_list["scene_number"].values
    frames = []
    for scene_number in scene_list:
        scene_number_str = str(scene_number + 1).zfill(3)
        key_frame_path = f"{key_frame_dir}/{scene_number_str}.jpg"
        if not os.path.exists(key_frame_path):
            print(f"Missing key frame images {key_frame_path}")
            return False
        frames.append(Image.open(key_frame_path))
    return frames


def pad_or_sample_frames(frames, target_frames=32, do_pad=True):
    """
    Adjusts the number of frames to match the required input size for ViViT.

    Args:
        frames (list of PIL.Image or numpy arrays): List of extracted frames.
        target_frames (int): Number of frames required (default: 32).

    Returns:
        list: A list of `target_frames` frames.
    """
    num_frames = len(frames)

    if num_frames < target_frames and do_pad:
        frames += [frames[-1]] * (target_frames - num_frames)
    elif num_frames > target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]

    return frames


def map_sequence_to_frames(seq_feats: torch.Tensor, num_frames: int = 32, tubelet_t: int = 2) -> torch.Tensor:
    """
    Maps a [sequence_len, 768] ViViT output to a [num_frames, 768] frame-level representation
    by averaging embeddings from all tubelets that cover each frame.

    Args:
        seq_feats (torch.Tensor): Shape [3137, 768] including CLS token.
        num_frames (int): Number of frames in the video (typically 32).
        tubelet_t (int): Temporal size of the tubelet (typically 2).

    Returns:
        torch.Tensor: Shape [num_frames, 768]
    """
    assert seq_feats.shape[0] == 3137, "Expected sequence length to be 3137 (including CLS token)."
    feat_dim = seq_feats.shape[1]

    tubelet_feats = seq_feats[1:]  # shape [3136, 768]

    num_t, num_h, num_w = 16, 14, 14  # from ViViT config
    tubelet_to_frame = defaultdict(list)

    for tubelet_idx in range(num_t * num_h * num_w):
        t_idx = tubelet_idx // (num_h * num_w)
        start_frame = t_idx * tubelet_t
        end_frame = min(start_frame + tubelet_t, num_frames)
        for f in range(start_frame, end_frame):
            tubelet_to_frame[f].append(tubelet_idx)

    frame_feats = []
    for f in range(num_frames):
        indices = tubelet_to_frame[f]
        if indices:
            frame_feat = tubelet_feats[indices].mean(dim=0)
        else:
            frame_feat = torch.zeros(feat_dim, device=seq_feats.device)
        frame_feats.append(frame_feat)

    return torch.stack(frame_feats)  # [num_frames, 768]


"""
Visual
"""


def extract_visual_features_vit(
        key_frame_dir, output_dir, visual_processor, visual_encoder,
        output_types_to_save=["full", "pooler_mean", "pooler_first"],
        device="cuda", verbose=False,
        batch_size=32):
    key_frames = load_key_frames(key_frame_dir)

    if not key_frames:
        return False

    if verbose:
        print(f"Loaded {len(key_frames)} key frames")

    seq_output = []
    for batch_start in range(0, len(key_frames), batch_size):
        batch = key_frames[batch_start:batch_start + batch_size]
        inputs = visual_processor(batch, return_tensors="pt").to(device)
        with torch.no_grad():
            output = visual_encoder(**inputs).pooler_output
            seq_output.append(output.cpu())
    seq_output = torch.cat(seq_output, dim=0)  # num_frames, 768
    if "full" in output_types_to_save:
        np.save(os.path.join(output_dir, "full.npy"), seq_output.numpy())
    if "pooler_mean" in output_types_to_save:
        np.save(os.path.join(output_dir, "pooler_mean.npy"), seq_output.mean(dim=0).numpy())
    return True


def extract_visual_features_vivit(key_frame_dir,
        output_dir,
        visual_processor, visual_encoder,
        output_types_to_save=["full", "pooler_mean", "pooler_first"],
        device="cuda", verbose=False):
    key_frames = load_key_frames(key_frame_dir)

    if not key_frames:
        return False

    # ViViT has the max number of frames set to 32
    if len(key_frames) != 32:
        key_frames = pad_or_sample_frames(key_frames, target_frames=32)

    if verbose:
        print(f"Loaded {len(key_frames)} key frames")

    input = visual_processor(key_frames, return_tensors="pt").to(device)
    output = visual_encoder(**input)
    print(f"Saving visual embeddings to {output_dir}")
    if verbose:
        print(f"Output shape: {output.last_hidden_state.shape}")
    if "full" in output_types_to_save:
        np.save(os.path.join(output_dir, "full.npy"), output.last_hidden_state.squeeze(dim=0).cpu().detach().numpy())
    if "pooler_mean" in output_types_to_save:
        np.save(os.path.join(output_dir, "pooler_mean.npy"), output.last_hidden_state.squeeze(dim=0).mean(dim=0).cpu().detach().numpy())
    if "pooler_first" in output_types_to_save:
        np.save(os.path.join(output_dir, "pooler_first.npy"), output.pooler_output.squeeze(dim=0).cpu().detach().numpy())
    return True


def extract_visual_features_llava_next(key_frame_dir, output_dir, visual_processor, visual_encoder,
    output_types_to_save=["full", "pooler_mean", "pooler_first"], device="cuda", verbose=False):
    key_frames = load_key_frames(key_frame_dir)

    if not key_frames:
        return False

    if verbose:
        print(f"Loaded {len(key_frames)} key frames")

    input = visual_processor(key_frames, return_tensors="pt")["pixel_values_videos"].to(device)  # B x num_frames x 3 x H x W
    print(input.shape)
    # this is with the projection to text (after multimodal projector)
    output = visual_encoder.get_video_features(input, vision_feature_layer=-2, vision_feature_select_strategy='default')[0]  # num_frames x 144 x 4096

    print(f"Saving visual embeddings to {output_dir}")
    if verbose:
        print(f"Output shape: {output.shape}")
    if "full" in output_types_to_save:
        np.save(os.path.join(output_dir, "full.npy"), output.cpu().detach().numpy())
    if "pooler_mean" in output_types_to_save:
        np.save(os.path.join(output_dir, "pooler_mean.npy"), output.mean(dim=(0, 1)).cpu().detach().numpy())  # (4096)
    return True


"""
Audio
"""


def extract_audio(video_path, stereo=False):
    output_dir = os.path.dirname(video_path)
    extracted = False
    if stereo:
        if not os.path.exists(f"{output_dir}/raw_stereo.wav"):
            os.system(f"ffmpeg -i {video_path} -ac 2 -ar 48000 -vn {output_dir}/raw_stereo.wav")
            extracted = True
        return f"{output_dir}/raw_stereo.wav", extracted
    else:
        if not os.path.exists(f"{output_dir}/audio.wav"):
            os.system(f"ffmpeg -i {video_path} -ar 48000 -vn {output_dir}/audio.wav")
            extracted = True
        return f"{output_dir}/audio.wav", extracted


def timecode_to_seconds(tc):
    h, m, s = map(float, tc.split(":"))
    return h * 3600 + m * 60 + s


def extract_audio_features_clap(audio_path, output_dir, audio_processor, audio_encoder,
                                output_types_to_save=["full", "pooler_mean"],
                                device="cuda", verbose=False, batch_size=8, scene_list=None):
    """
    Extracts CLAP audio features from either fixed-length chunks or scene-based segments.

    Args:
        scene_list (list): Optional list of scenes, each with keys:
            'start_time', 'end_time' in HH:MM:SS.xx format
    """

    SAMPLE_RATE = 48000
    MAX_SAMPLES = SAMPLE_RATE * 10

    os.makedirs(output_dir, exist_ok=True)

    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy()  # Convert to mono
    duration_sec = len(waveform) / SAMPLE_RATE

    if verbose:
        print(f"Loaded audio duration: {duration_sec:.2f}s")

    segments = []

    if scene_list:
        for scene in scene_list:
            start_sec = timecode_to_seconds(scene['start_time'])
            end_sec = timecode_to_seconds(scene['end_time'])
            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample = int(end_sec * SAMPLE_RATE)
            segment = waveform[start_sample:end_sample]
            segments.append(segment)
    else:
        num_chunks = int(np.ceil(len(waveform) / MAX_SAMPLES))
        segments = [waveform[i * MAX_SAMPLES: (i + 1) * MAX_SAMPLES] for i in range(num_chunks)]

    pooler_first_full = []

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        max_len = max(len(x) for x in batch)
        padded_batch = [np.pad(x, (0, max_len - len(x))) for x in batch]

        inputs = audio_processor(audios=padded_batch, return_tensors="pt", sampling_rate=SAMPLE_RATE, padding=True).to(device)

        with torch.no_grad():
            output = audio_encoder(**inputs, output_hidden_states=False)

        pooler_first_full.append(output.pooler_output)  # B, 512

        if verbose:
            print(f"Processed batch {batch_start // batch_size + 1}")

    full_concat = torch.cat(pooler_first_full, dim=0).cpu()  # num_scenes, 512
    if "full" in output_types_to_save:
        np.save(os.path.join(output_dir, "full.npy"), full_concat.numpy())

    if "pooler_mean" in output_types_to_save:
        mean_embedding = full_concat.mean(dim=0).numpy()  # 512
        np.save(os.path.join(output_dir, "pooler_mean.npy"), mean_embedding)

    if verbose:
        print(f"Saved features to {output_dir}")

    return True


def extract_audio_features_hubert(audio_path, output_dir, audio_processor, audio_encoder,
                                  output_types_to_save=["full", "pooler_mean"],
                                  device="cuda", verbose=False, batch_size=8, scene_list=None):
    """
    Extracts HuBERT audio features from an audio file based on scenes or chunks.

    Args:
        scene_list (list): Optional list of dicts with keys 'start_time' and 'end_time' in HH:MM:SS.xx format.
    """

    SAMPLE_RATE = 16000
    MAX_SAMPLES = SAMPLE_RATE * 10  # 10-second chunks

    os.makedirs(output_dir, exist_ok=True)

    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform).mean(dim=0).numpy()  # mono & 16kHz

    duration_sec = len(waveform) / SAMPLE_RATE
    if verbose:
        print(f"Audio loaded: {duration_sec:.2f}s")

    segments = []

    if scene_list:
        for scene in scene_list:
            start_sec = timecode_to_seconds(scene['start_time'])
            end_sec = timecode_to_seconds(scene['end_time'])
            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample = int(end_sec * SAMPLE_RATE)
            if end_sample - start_sample > MAX_SAMPLES:
                end_sample = start_sample + MAX_SAMPLES
            segments.append(waveform[start_sample:end_sample])
    else:
        num_chunks = int(np.ceil(len(waveform) / MAX_SAMPLES))
        segments = [waveform[i * MAX_SAMPLES: (i + 1) * MAX_SAMPLES] for i in range(num_chunks)]

    batch_hidden_states = []

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        max_len = max(len(seg) for seg in batch)
        padded_batch = [np.pad(seg, (0, max_len - len(seg))) for seg in batch]

        inputs = audio_processor(padded_batch, return_tensors="pt", sampling_rate=SAMPLE_RATE, padding=True).to(device)

        with torch.no_grad():
            outputs = audio_encoder(**inputs)
            batch_hidden_states.append(outputs.last_hidden_state.mean(dim=1))  # B, D
            print(f"Processed batch {batch_start // batch_size + 1}")
    batch_hidden_states = torch.cat(batch_hidden_states, dim=0).cpu().numpy()  # num_scenes, D

    if "full" in output_types_to_save:
        np.save(os.path.join(output_dir, "full.npy"), batch_hidden_states)

    if "pooler_mean" in output_types_to_save:
        mean_embedding = batch_hidden_states.mean(axis=0)  # D
        np.save(os.path.join(output_dir, "pooler_mean.npy"), mean_embedding)

    if verbose:
        print(f"Saved HuBERT features to {output_dir}")

    return True


def main(args):
    duration_data = None
    video_ids_with_duration = []
    if args.duration_csv:
        print(f"Loading duration data from {args.duration_csv}")
        duration_data = pd.read_csv(args.duration_csv).set_index('videoid')
        video_ids_with_duration = duration_data.index

    to_be_processed_video_ids = []
    if args.video_id:
        to_be_processed_video_ids.append(args.video_id)
    elif args.video_id_file:
        to_be_processed_video_ids = open(args.video_id_file).read().splitlines()
    else:
        raise ValueError("Either video_id or video_id_file should be provided")
    print(f"Processing {len(to_be_processed_video_ids)} videos in total")

    processed_videos = 0

    visual_processor = None
    visual_encoder = None
    if args.extract_visual_features:
        model_path = VISION_ENCODER_ZOO[args.visual_backbone]["model_path"]
        if args.visual_backbone == "vivit":
            print(f"Loading pretrained visual encoder for {args.visual_backbone}")
            from transformers import VivitModel, VivitImageProcessor
            visual_processor = VivitImageProcessor.from_pretrained(model_path)
            visual_encoder = VivitModel.from_pretrained(model_path,
                                            attn_implementation="sdpa",
                                            ).to("cuda")

        elif args.visual_backbone == "llava_next":
            print(f"Loading pretrained visual encoder for {args.visual_backbone}")
            from transformers import AutoProcessor, AutoModelForPreTraining
            visual_processor = AutoProcessor.from_pretrained(model_path).video_processor
            visual_encoder = AutoModelForPreTraining.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

        elif args.visual_backbone == "vit":
            print(f"Loading pretrained visual encoder for {args.visual_backbone}")
            from transformers import ViTModel, ViTImageProcessor
            visual_processor = ViTImageProcessor.from_pretrained(model_path)
            visual_encoder = ViTModel.from_pretrained(model_path).to("cuda")

        else:
            raise NotImplementedError(f"Visual backbone {args.visual_backbone} not implemented")

    audio_processor = None
    audio_encoder = None
    if args.extract_audio_features:
        model_path = AUDIO_ENCODER_ZOO[args.audio_backbone]["model_path"]
        if args.audio_backbone == "hubert_large":
            print(f"Loading pretrained audio encoder for {args.audio_backbone}")
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            audio_encoder = HubertModel.from_pretrained(model_path).to("cuda")
        else:
            from transformers import ClapAudioModel, ClapProcessor
            audio_processor = ClapProcessor.from_pretrained(model_path)
            audio_encoder = ClapAudioModel.from_pretrained(model_path).to("cuda")

    for video_id in tqdm(to_be_processed_video_ids):
        print(f"Processing video {video_id}")
        video_input_dir = os.path.join(args.raw_video_dir, video_id)
        if not os.path.exists(os.path.join(video_input_dir, "raw.mp4")) and not os.path.exists(os.path.join(video_input_dir, "raw_trimmed.mp4")):
            print(f"Video {video_id} does not exist in {video_input_dir} so cannot be processed")
            continue
        video_path = os.path.join(args.raw_video_dir, video_id, "raw.mp4")
        # Trim video
        video_duration = None
        if video_id in video_ids_with_duration:
            video_duration = duration_data.loc[video_id]['duration']
        video_path = trim_video_outro(video_path, video_duration_s=video_duration, remove_untrimmed=False)
        if not os.path.exists(video_path):
            print(f"Video {video_id} does not exist in {video_path} so cannot be processed")
            continue
        # Extract key frames
        if args.extract_key_frames:
            keyframe_dir = os.path.join(args.key_frame_dir, video_id)
            if os.path.exists(os.path.join(keyframe_dir, "index.csv")) and not args.reprocess:
                print(f"Key frames for video {video_id} already extracted so skip key frame extraction")
                try:
                    key_frame_list = pd.read_csv(os.path.join(keyframe_dir, "index.csv"))
                except pd.errors.EmptyDataError:
                    print(f"Empty key frame index file {os.path.join(keyframe_dir, 'index.csv')}")
                    print(f"Retry")
                    key_frame_list = extract_keyframes(video_path, keyframe_dir)
            else:
                print(f"Extracting key frames for video {video_id}")
                extract_keyframes(video_path, keyframe_dir)
                processed_videos += 1

        # Extract visual features
        if args.extract_visual_features:
            keyframe_dir = os.path.join(args.key_frame_dir, video_id)
            visual_feature_dir = os.path.join(args.processed_feature_dir, "visual", f"{args.visual_backbone}", video_id)
            os.makedirs(visual_feature_dir, exist_ok=True)
            if not os.path.exists(keyframe_dir):
                print(f"Key frames for video {video_id} does not exist so skip visual feature extraction")

            visual_feature_types = ["full", "pooler_mean"]
            if all([os.path.exists(os.path.join(visual_feature_dir, f"{o}.npy")) for o in visual_feature_types]) and not args.reprocess:
                print(f"Visual features for video {video_id} already extracted so skip visual feature extraction")
            else:
                print(f"Extracting visual features for video {video_id}")
                if args.visual_backbone == "vivit":
                    extracted = extract_visual_features_vivit(keyframe_dir, visual_feature_dir, visual_processor, visual_encoder,
                    output_types_to_save=visual_feature_types, device="cuda", verbose=True)
                elif args.visual_backbone == "llava_next":
                    extracted = extract_visual_features_llava_next(keyframe_dir, visual_feature_dir, visual_processor, visual_encoder,
                    output_types_to_save=visual_feature_types, device="cuda", verbose=True)
                elif args.visual_backbone == "vit":
                    extracted = extract_visual_features_vit(keyframe_dir, visual_feature_dir, visual_processor, visual_encoder,
                    output_types_to_save=visual_feature_types, device="cuda", verbose=True, batch_size=128)
                else:
                    raise NotImplementedError(f"Visual backbone {args.visual_backbone} not implemented")
                if extracted:
                    processed_videos += 1

        # Extract audio
        if args.extract_audio:
            audio_path, extracted = extract_audio(video_path, stereo=False)
            if extracted:
                processed_videos += 1

        # Extract audio features
        if args.extract_audio_features:
            if not os.path.exists(os.path.join(args.key_frame_dir, video_id, "index.csv")):
                print(f"Key scenes for video {video_id} do not exist so skip audio feature extraction")
                continue
            try:
                key_scene_list = pd.read_csv(os.path.join(args.key_frame_dir, video_id, "index.csv"))
            except pd.errors.EmptyDataError:
                print(f"Empty key frame index file {os.path.join(args.key_frame_dir, video_id, 'index.csv')}")
                continue
            key_scene_list = key_scene_list.to_dict(orient="records")
            audio_path, _ = extract_audio(video_path, stereo=False)
            audio_feature_dir = os.path.join(args.processed_feature_dir, "audio", f"{args.audio_backbone}", video_id)
            os.makedirs(audio_feature_dir, exist_ok=True)
            if not os.path.exists(audio_path):
                print(f"Audio for video {video_id} does not exist so skip audio feature extraction")
            audio_feature_types = ["full", "pooler_mean"]  # (num_scenes, D) ; (D,)
            if all([os.path.exists(os.path.join(audio_feature_dir, f"{o}.npy")) for o in audio_feature_types]) and not args.reprocess:
                print(f"Audio features for video {video_id} already extracted so skip audio feature extraction")
            else:
                try:
                    print(f"Extracting audio features for video {video_id}")
                    if args.audio_backbone == "hubert_large":
                        extracted = extract_audio_features_hubert(audio_path, audio_feature_dir, audio_processor, audio_encoder,
                        output_types_to_save=audio_feature_types, device="cuda", verbose=False,
                        batch_size=64, scene_list=key_scene_list)
                    else:
                        extracted = extract_audio_features_clap(audio_path, audio_feature_dir, audio_processor, audio_encoder,
                        output_types_to_save=audio_feature_types, device="cuda", verbose=False,
                        batch_size=64, scene_list=key_scene_list)
                    if extracted:
                        processed_videos += 1
                except Exception as e:
                    print(f"Error extracting audio features for video {video_id}: {e}")
                    continue

        if args.max_videos and processed_videos >= args.max_videos:
            print(f"Reached maximum number of videos to process so stopped")
            break

    print(f"Processed {processed_videos} videos")


# Example usage (run from repo root):
#   python data_preprocessing/process_video.py --raw_video_dir data/raw_video \
#       --key_frame_dir data/processed_data/key_frames --processed_feature_dir data/processed_data/features \
#       --extract_key_frames --extract_visual_features --extract_audio --extract_audio_features \
#       --visual_backbone vit --audio_backbone clap_general --video_id_file data/video_ids.txt
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process video')
    # Input
    parser.add_argument('--raw_video_dir', type=str, help='Directory containing raw videos')
    parser.add_argument('--video_id', type=str, help='Path to raw video')
    parser.add_argument('--video_id_file', type=str, help='Path to list of video ids')
    parser.add_argument('--key_frame_dir', type=str, help='Output directory')
    parser.add_argument('--processed_feature_dir', type=str, help='Output directory')
    parser.add_argument('--duration_csv', type=str, default=None,
                         help='Optional CSV (indexed by videoid, with a duration column) used to trim known channel-ad outros before processing')
    parser.add_argument('--visual_backbone', type=str, default='vivit')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess video')
    parser.add_argument('--max_videos', type=int, default=None, help='Maximum number of videos to process')
    # Task arguments
    parser.add_argument('--extract_key_frames', action='store_true', help='Extract key frames')
    parser.add_argument('--extract_visual_features', action='store_true', help='Extract visual features')
    parser.add_argument('--extract_audio', action='store_true', help='Extract audio')
    parser.add_argument('--extract_audio_features', action='store_true', help='Extract audio features')
    # Audio arguments
    parser.add_argument('--audio_backbone', type=str, default='clap', help='Audio backbone')
    args = parser.parse_args()

    os.makedirs(args.processed_feature_dir, exist_ok=True)
    main(args)
