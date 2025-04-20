import cv2
import numpy as np
import os
# from skimage.metrics import structural_similarity as ssim
# from statistics import mean
import pandas as pd


def extract_frames(video_path, output_dir, fps=30, start_time=1, end_time=4):
    """
    Extract frames from a specific time window in the video

    Parameters:
    - fps: frames per second to extract
    - start_time: start time in seconds
    - end_time: end time in seconds
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    # Calculate frame indices for start and end times
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    frame_count = 0
    extracted_frame_count = 0

    while True:
        ret, frame = cap.read()

        # Stop if no more frames
        if not ret:
            break

        # Skip frames before start time
        if frame_count < start_frame:
            frame_count += 1
            continue

        # Stop if reached end time
        if frame_count > end_frame:
            break

        # Extract and save frame
        frame_path = os.path.join(output_dir, f"frame_{extracted_frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        extracted_frame_count += 1
        frame_count += 1

    cap.release()
    return extracted_frame_count


def detect_swing_sequence(input_dir, output_dir, num_frames=15):
    """Detect and select frames showing the batter's swing sequence without prior knowledge of start/end frames,
    focusing only on the first half of the available frames"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    frame_files.sort()

    frames = []

    for frame_file in frame_files:
        frame_path = os.path.join(input_dir, frame_file)
        frame = cv2.imread(frame_path)
        frames.append((frame_path, frame))

    # Continue with the remaining function as before, now operating only on the first half of frames
    # Calculate frame-to-frame differences
    frame_diffs = []
    for i in range(1, len(frames)):
        prev_gray = cv2.cvtColor(frames[i - 1][1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i][1], cv2.COLOR_BGR2GRAY)

        # Focus on lower half of frame where batter likely is
        h, w = prev_gray.shape
        prev_gray_lower = prev_gray[h // 2:, :]
        curr_gray_lower = curr_gray[h // 2:, :]

        # Calculate absolute difference between frames
        diff = cv2.absdiff(prev_gray_lower, curr_gray_lower)
        diff_score = np.sum(diff) / (diff.shape[0] * diff.shape[1])
        frame_diffs.append((i, diff_score))

    # Smooth the differences to reduce noise
    window_size = 5
    smoothed_diffs = []
    for i in range(len(frame_diffs)):
        start = max(0, i - window_size // 2)
        end = min(len(frame_diffs), i + window_size // 2 + 1)
        window = [frame_diffs[j][1] for j in range(start, end)]
        smoothed_diff = sum(window) / len(window)
        smoothed_diffs.append((frame_diffs[i][0], smoothed_diff))

    # Calculate the gradient (rate of change) of differences
    diff_gradients = []
    for i in range(1, len(smoothed_diffs)):
        curr_idx, curr_diff = smoothed_diffs[i]
        prev_idx, prev_diff = smoothed_diffs[i - 1]
        gradient = curr_diff - prev_diff
        diff_gradients.append((curr_idx, gradient))

    # Find significant motion events
    # A swing typically has a sharp increase in motion followed by a peak and then decrease
    mean_gradient = np.mean([abs(g) for _, g in diff_gradients])
    std_gradient = np.std([g for _, g in diff_gradients])
    threshold = mean_gradient + std_gradient

    # Find regions of significant motion (potential swings)
    significant_motion_starts = []
    for i, (idx, gradient) in enumerate(diff_gradients):
        if gradient > threshold:
            # Check if this is the start of a new significant motion
            if not significant_motion_starts or idx > significant_motion_starts[-1][0] + 5:
                significant_motion_starts.append((idx, i))

    # For each significant motion start, find the corresponding peak and end
    swing_candidates = []
    for start_idx, gradient_idx in significant_motion_starts:
        # Find the peak after this start
        peak_idx = start_idx
        peak_diff = smoothed_diffs[gradient_idx][1]

        for j in range(gradient_idx, min(len(smoothed_diffs), gradient_idx + 20)):
            if smoothed_diffs[j][1] > peak_diff:
                peak_diff = smoothed_diffs[j][1]
                peak_idx = smoothed_diffs[j][0]

        # Find where the motion subsides (end of swing)
        end_idx = peak_idx
        for j in range(peak_idx - start_idx + gradient_idx, min(len(smoothed_diffs), gradient_idx + 40)):
            if smoothed_diffs[j][1] < peak_diff * 0.3:  # Motion reduced significantly
                end_idx = smoothed_diffs[j][0]
                break

        # Calculate swing duration and intensity
        duration = end_idx - start_idx
        intensity = peak_diff

        # Store this swing candidate
        swing_candidates.append((start_idx, peak_idx, end_idx, intensity, duration))

    # If no significant motions found, fall back to the highest motion frames
    if not swing_candidates:
        sorted_diffs = sorted(smoothed_diffs, key=lambda x: x[1], reverse=True)
        center_idx = sorted_diffs[0][0]
        # Estimate a reasonable swing duration
        start_idx = max(1, center_idx - 15)
        end_idx = min(len(frames) - 1, center_idx + 15)
        swing_candidates.append((start_idx, center_idx, end_idx, sorted_diffs[0][1], end_idx - start_idx))

    # Select the most likely swing based on intensity and duration
    # MLB swings typically last about 0.5-0.75 seconds (15-22 frames at 30fps)
    # We'll score candidates based on both intensity and appropriate duration
    best_candidate = None
    best_score = -1

    for start_idx, peak_idx, end_idx, intensity, duration in swing_candidates:
        # Score based on intensity and duration
        duration_score = 1.0
        if duration < 10:  # Too short to be a full swing
            duration_score = duration / 10
        elif duration > 40:  # Too long for a single swing
            duration_score = 40 / duration

        score = intensity * duration_score

        if score > best_score:
            best_score = score
            best_candidate = (start_idx, peak_idx, end_idx)

    if best_candidate:
        start_idx, peak_idx, end_idx = best_candidate

        # Take more frames before the start to capture wind-up
        extended_start = max(1, start_idx - 10)

        # Extend after the end to capture follow-through
        extended_end = min(len(frames) - 1, end_idx + 5)

        # Create a sequence with the full swing
        swing_sequence_idxs = list(range(extended_start, extended_end + 1))

        # Ensure we have the right number of frames
        if len(swing_sequence_idxs) > num_frames:
            # Sample the sequence to get the desired number of frames
            # Keep more frames around the peak of the swing
            peak_relative_idx = peak_idx - extended_start

            if peak_relative_idx < 0:
                peak_relative_idx = 0
            elif peak_relative_idx >= len(swing_sequence_idxs):
                peak_relative_idx = len(swing_sequence_idxs) - 1

            # Calculate importance weights (higher near the peak)
            weights = [1.0 / (1.0 + 0.2 * abs(i - peak_relative_idx)) for i in range(len(swing_sequence_idxs))]

            # Normalize weights to probabilities
            weights_sum = sum(weights)
            probs = [w / weights_sum for w in weights]

            # Select frames with probability proportional to weights
            # but ensure we include start, peak, and end frames
            must_include = [0, peak_relative_idx, len(swing_sequence_idxs) - 1]
            remaining = num_frames - len(must_include)

            # Set probability of must-include indices to 0 to avoid duplicates
            for idx in must_include:
                if 0 <= idx < len(probs):
                    probs[idx] = 0

            # Renormalize probabilities
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]

                # Sample remaining frames
                remaining_indices = np.random.choice(
                    range(len(swing_sequence_idxs)),
                    size=min(remaining, len(swing_sequence_idxs) - len(must_include)),
                    replace=False,
                    p=probs
                ).tolist()
            else:
                # Fallback if all probabilities are 0
                remaining_indices = []

            # Combine must-include and sampled indices
            selected_relative_indices = must_include + remaining_indices
            selected_indices = [swing_sequence_idxs[i] for i in selected_relative_indices if
                                i < len(swing_sequence_idxs)]

            # If we still need more frames, add them evenly
            if len(selected_indices) < num_frames:
                total_frames = end_idx - extended_start + 1
                step = max(1, total_frames // (num_frames - len(selected_indices)))
                additional_indices = list(range(extended_start, end_idx + 1, step))

                # Remove indices we already have
                additional_indices = [idx for idx in additional_indices if idx not in selected_indices]

                # Add as many as needed
                selected_indices.extend(additional_indices[:num_frames - len(selected_indices)])

            # Sort indices to maintain chronological order
            selected_indices.sort()

            # Truncate to the requested number
            selected_indices = selected_indices[:num_frames]
        else:
            # If we have fewer frames than requested, use all of them
            selected_indices = swing_sequence_idxs

            # If we need more frames, add frames from before and after the sequence
            additional_needed = num_frames - len(selected_indices)
            if additional_needed > 0:
                # Add frames before
                before_frames = list(range(max(1, extended_start - additional_needed // 2), extended_start))
                # Add frames after
                after_frames = list(
                    range(extended_end + 1, min(len(frames), extended_end + 1 + additional_needed // 2)))
                # Combine
                additional_frames = before_frames + after_frames
                # Sort and take what we need
                additional_frames.sort()
                selected_indices.extend(additional_frames[:additional_needed])
                selected_indices.sort()
    else:
        # Fallback: select evenly spaced frames from the entire video
        step = max(1, len(frames) // num_frames)
        selected_indices = list(range(1, len(frames), step))[:num_frames]

    if len(selected_indices) == 60:
        # If exactly 60 frames, take first 30
        selected_indices = selected_indices[:30]
    if 60 > len(selected_indices) > 50:
        selected_indices = selected_indices[:26]
    elif len(selected_indices) < 50:
        # If less than 50 frames, take first 25
        selected_indices = selected_indices[:25]

    # Get the selected frames
    swing_sequence = [frames[idx][0] for idx in selected_indices if 0 <= idx < len(frames)]

    # Return the swing sequence paths for further processing
    return swing_sequence


def process_and_enhance_frames(frame_paths, output_dir, video_source='default', crop_dim=(640, 640),
                               final_dim=(256, 256),
                               quality_enhance=True, interpolation=cv2.INTER_LANCZOS4, sharpen_sigma=0.5):
    """
    Process selected frames with center cropping, resizing and quality enhancement.

    Parameters:
    frame_paths (list): List of paths to input frames
    output_dir (str): Directory to save processed frames
    crop_dim (tuple): Dimensions for initial cropping (width, height)
    final_dim (tuple): Final dimensions after resizing (width, height)
    quality_enhance (bool): Whether to apply quality enhancement
    interpolation (int): OpenCV interpolation method
    sharpen_sigma (float): Strength of sharpening (0.0-2.0, higher = stronger)

    Returns:
    list: Paths to processed frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_frames = []

    for i, frame_path in enumerate(frame_paths):
        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to read frame: {frame_path}")
            continue

        # Apply bilateral filter to reduce noise while preserving edges
        if quality_enhance:
            frame = cv2.bilateralFilter(frame, 9, 75, 75)

        # Center crop
        h, w = frame.shape[:2]

        # Calculate center point
        center_x = w // 2
        center_y = h // 2

        # Calculate crop coordinates
        crop_width, crop_height = crop_dim

        # Ensure crop dimensions don't exceed image dimensions
        crop_width = min(crop_width, w)
        crop_height = min(crop_height, h)

        # Calculate coordinates for center crop
        start_x = max(0, center_x - crop_width // 2)
        start_y = max(0, center_y - crop_height // 2)
        end_x = min(w, start_x + crop_width)
        end_y = min(h, start_y + crop_height)

        # Adjust if we hit the boundary
        if end_x - start_x < crop_width:
            start_x = max(0, end_x - crop_width)
        if end_y - start_y < crop_height:
            start_y = max(0, end_y - crop_height)

        # Crop to target dimensions
        cropped_frame = frame[start_y:end_y, start_x:end_x]

        # If the cropped dimensions aren't exact, resize to exact crop dimensions
        if cropped_frame.shape[0] != crop_height or cropped_frame.shape[1] != crop_width:
            cropped_frame = cv2.resize(cropped_frame, crop_dim, interpolation=interpolation)

        # Now resize to final dimensions
        final_frame = cv2.resize(cropped_frame, final_dim, interpolation=interpolation)

        # Apply sharpening if enabled (after all resizing to preserve details)
        if quality_enhance and sharpen_sigma > 0:
            # Create sharpening kernel
            blur = cv2.GaussianBlur(final_frame, (0, 0), sharpen_sigma)
            final_frame = cv2.addWeighted(final_frame, 1.5, blur, -0.5, 0)

        # Generate a unique output filename using the index and original filename
        frame_name = os.path.basename(frame_path)
        file_base, file_ext = os.path.splitext(frame_name)

        unique_id = f"{video_source}_frame_{i:04d}{file_ext}"

        output_path = os.path.join(output_dir, unique_id)

        # Save frame with high quality
        cv2.imwrite(output_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        processed_frames.append(output_path)

    return processed_frames


def process_video(video_path, output_base_dir, crop_dim=(640, 640), final_dim=(256, 256), fps=30, start_time=3.66,
                  end_time=4):
    """
    Process video focusing on 3-4 second window
    """
    # Extract video filename (without extension) to use as source identifier
    video_source = os.path.splitext(os.path.basename(video_path))[0]

    # Create output directories
    output_dirs = {
        'extracted_frames': os.path.join(output_base_dir, "extracted_frames"),
        'swing_frames': os.path.join(output_base_dir, "swing_frames"),
        'processed_swing': os.path.join(output_base_dir, "processed_swing")
    }

    # Create directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Step 1: Extract frames from 1-4 second window
    print("Extracting frames from specified video segment...")
    extract_frames(
        video_path,
        output_dirs['extracted_frames'],
        fps=fps,
        start_time=start_time,
        end_time=end_time
    )

    # Step 2: Detect swing sequence in extracted frames
    print("Detecting swing sequence...")
    swing_paths = detect_swing_sequence(
        output_dirs['extracted_frames'],
        output_dirs['swing_frames'],
        num_frames=60
    )

    # Step 3: Process and enhance swing frames
    print("Processing swing frames...")
    processed_swing_paths = process_and_enhance_frames(
        swing_paths,
        output_dirs['processed_swing'],
        video_source=video_source,  # Pass video source identifier
        crop_dim=(640, 640),
        final_dim=(178, 218),
        quality_enhance=True
    )

    return {
        "raw_frames": output_dirs['extracted_frames'],
        "swing_frames": output_dirs['swing_frames'],
        "processed_swing": processed_swing_paths
    }


if __name__ == "__main__":
    df = pd.read_csv('2024-mlb-homeruns.csv')
    limit =

    for i, video_path in enumerate(df['video'].head(limit), start=1):
        print(f"Processing video {i}/{limit}: {video_path}")

        output_base_dir = "baseball_video_processing"

        # Process the video with high-quality cropping and resizing
        results = process_video(
            video_path,
            output_base_dir,
            crop_dim=(640, 640),  # Initial crop dimensions
            final_dim=(178, 218)  # Final resize dimensions
        )

        print("Processing results:")
        for category, paths in results.items():
            if isinstance(paths, list):
                print(f"- {category}: {len(paths)} frames")
            else:
                print(f"- {category}: {paths}")
