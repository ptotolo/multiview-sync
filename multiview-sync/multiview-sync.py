import math
import numpy as np
import subprocess
import scipy.io.wavfile
import os 

def safe_delete(file_path):
  """
  Attempts to delete a file, ignoring errors if the file doesn't exist.

  Args:
    file_path: The path to the file to be deleted.
  """
  try:
    os.remove(file_path)
    print(f"File '{file_path}' deleted successfully.")
  except FileNotFoundError as e:
    print(f"File '{file_path}' not found. Ignoring.")

# Extract audio from video file, save as wav audio file
# INPUT: Video file
# OUTPUT: Returns the path to the saved audio file
def extract_audio(dir, video_file):
    track_name = video_file.split(".")
    audio_output = track_name[0] + ".wav"  # !! CHECK TO SEE IF FILE IS IN UPLOADS DIRECTORY
    output = dir + audio_output
    subprocess.call(["ffmpeg", "-y", "-i", dir + video_file, "-vn", "-ac", "1", "-f", "wav", output])
    return output

# Read file
# INPUT: Audio file
# OUTPUT: Sets sample rate of wav file, Returns data read from wav file (numpy array of integers)
def read_audio(audio_file):
    rate, data = scipy.io.wavfile.read(audio_file)  # Return the sample rate (in samples/sec) and data from a WAV file
    return data, rate


def make_horiz_bins(data, fft_bin_size, overlap, box_height):
    horiz_bins = {}
    # process first sample and set matrix height
    sample_data = data[0:fft_bin_size]  # get data for first sample
    if len(sample_data) == fft_bin_size:  # if there are enough audio points left to create a full fft bin
        intensities = fourier(sample_data)  # intensities is list of fft results
        for i in range(len(intensities)):
            box_y = i // box_height
            if box_y in horiz_bins:
                horiz_bins[box_y].append((intensities[i], 0, i))  # (intensity, x, y)
            else:
                horiz_bins[box_y] = [(intensities[i], 0, i)]
    # process remainder of samples
    x_coord_counter = 1  # starting at second sample, with x index 1
    for j in range(int(fft_bin_size - overlap), len(data), int(fft_bin_size - overlap)):
        sample_data = data[j:j + fft_bin_size]
        if len(sample_data) == fft_bin_size:
            intensities = fourier(sample_data)
            for k in range(len(intensities)):
                box_y = k // box_height
                if box_y in horiz_bins:
                    horiz_bins[box_y].append((intensities[k], x_coord_counter, k))  # (intensity, x, y)
                else:
                    horiz_bins[box_y] = [(intensities[k], x_coord_counter, k)]
        x_coord_counter += 1

    return horiz_bins


# Compute the one-dimensional discrete Fourier Transform
# INPUT: list with length of number of samples per second
# OUTPUT: list of real values len of num samples per second
def fourier(sample):
    mag = []
    fft_data = np.fft.fft(sample)  # Returns real and complex value pairs
    for i in range(len(fft_data) // 2):
        r = fft_data[i].real ** 2
        j = fft_data[i].imag ** 2
        mag.append(round(math.sqrt(r + j), 2))

    return mag


def make_vert_bins(horiz_bins, box_width):
    boxes = {}
    for key in horiz_bins.keys():
        for i in range(len(horiz_bins[key])):
            box_x = horiz_bins[key][i][1] // box_width
            if (box_x, key) in boxes:
                boxes[(box_x, key)].append((horiz_bins[key][i]))
            else:
                boxes[(box_x, key)] = [(horiz_bins[key][i])]

    return boxes


def find_bin_max(boxes, maxes_per_box):
    freqs_dict = {}
    for key in boxes.keys():
        max_intensities = [(1, 2, 3)]
        for i in range(len(boxes[key])):
            if boxes[key][i][0] > min(max_intensities)[0]:
                if len(max_intensities) < maxes_per_box:  # add if < number of points per box
                    max_intensities.append(boxes[key][i])
                else:  # else add new number and remove min
                    max_intensities.append(boxes[key][i])
                    max_intensities.remove(min(max_intensities))
        for j in range(len(max_intensities)):
            if max_intensities[j][2] in freqs_dict:
                freqs_dict[max_intensities[j][2]].append(max_intensities[j][1])
            else:
                freqs_dict[max_intensities[j][2]] = [max_intensities[j][1]]

    return freqs_dict

def find_freq_pairs(freqs_dict_orig, freqs_dict_sample):
    time_pairs = []
    for key in freqs_dict_sample.keys():  # iterate through freqs in sample
        if key in freqs_dict_orig:  # if same sample occurs in base
            for i in range(len(freqs_dict_sample[key])):  # determine time offset
                for j in range(len(freqs_dict_orig[key])):
                    time_pairs.append((freqs_dict_sample[key][i], freqs_dict_orig[key][j]))

    return time_pairs

def find_delay(time_pairs):
    t_diffs = {}
    for i in range(len(time_pairs)):
        delta_t = time_pairs[i][0] - time_pairs[i][1]
        if delta_t in t_diffs:
            t_diffs[delta_t] += 1
        else:
            t_diffs[delta_t] = 1
    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
    #print(t_diffs_sorted)
    time_delay = t_diffs_sorted[-1][0]

    return time_delay

# Find time delay between two video files
def align(video1, video2, dir, fft_bin_size=1024, overlap=0, box_height=512, box_width=43, samples_per_box=7):

    # Process first file
    wavfile1 = extract_audio(dir, video1)
    raw_audio1, rate = read_audio(wavfile1)
    bins_dict1 = make_horiz_bins(raw_audio1[:44100 * 240], fft_bin_size, overlap, box_height)  # bins, overlap, box height
    boxes1 = make_vert_bins(bins_dict1, box_width)  # box width
    ft_dict1 = find_bin_max(boxes1, samples_per_box)  # samples per box

    # Process second file
    wavfile2 = extract_audio(dir, video2)
    raw_audio2, rate = read_audio(wavfile2)
    bins_dict2 = make_horiz_bins(raw_audio2[:48000 * 240], fft_bin_size, overlap, box_height)
    boxes2 = make_vert_bins(bins_dict2, box_width)
    ft_dict2 = find_bin_max(boxes2, samples_per_box)

    # Determine time delay
    pairs = find_freq_pairs(ft_dict1, ft_dict2)
    delay = find_delay(pairs)
    samples_per_sec = float(rate) / float(fft_bin_size)
    seconds = round(float(delay) / float(samples_per_sec), 4)

    return seconds

def adjust_frame_rate(input_video, output_video, target_fps=30):
  """
  Adjusts the frame rate of a video using ffmpeg.

  Args:
      input_video: Path to the input video file.
      output_video: Path to the output video file with adjusted frame rate.
      target_fps: The desired target frame rate (default: 30 fps).
  """
  command = ["ffmpeg", "-i", input_video, "-c:v", "libx264", "-r", str(target_fps), output_video]
  subprocess.run(command, check=True)

def merge_videos_grid(video_path1, video_path2, video_path3, output_path, delay_seconds):
  """
  Merges three videos to fit in a square, each filling a quarter, with the lower right space staying empty, with audio and delays for the videos using ffmpeg. Trimming is applied to ensure equal length.

  Args:
      video_path1 (str): Path to the first video file.
      video_path2 (str): Path to the second video file.
      video_path3 (str): Path to the third video file.
      output_path (str): Path to the output video file.
      delay_seconds (tuple): The delay in seconds to apply to each video (delay1, delay2, delay3).
  """

  # Get video durations
  video_durations = {}
  for video_path, label in [(video_path1, "v1"), (video_path2, "v2"), (video_path3, "v3")]:
    duration_command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", video_path]
    output = subprocess.run(duration_command, capture_output=True, text=True, check=True)
    duration_info = output.stdout.split("\n")
    for line in duration_info:
      if "duration=" in line:
        video_durations[label] = float(line.split("=")[1])

  # Calculate base dimension based on shortest duration and desired grid layout
  shortest_duration = min(video_durations.values())
  
  # Apply delays and calculate trimming parameters
  #trim_start = []
  #for delay in delay_seconds.values():
  #  trim_start.append(delay)
  #for label, duration in video_durations.items():
  #  video_durations[label] = duration - delay_seconds[label]

  # Determine trimming based on shortest duration
  shortest_duration = min(video_durations.values())
  for label in video_durations.keys():
    video_durations[label] = (delay_seconds[label], shortest_duration + delay_seconds[label])

  # Generate ffmpeg command for merging with trimming and scaling
  command = [
      "ffmpeg",
      # First video (top left)
      "-ss", str(video_durations["v1"][0]),  # Start trimming
      "-t", str(shortest_duration),
      "-i", video_path1,
      #"-to", str(video_durations["v1"][1]),  # End trimming
      #"-filter_complex", f"[0:v]scale={base_dim0}:{base_dim1}[v1]",  # Scale video
      # Second video (top right)
      "-ss", str(video_durations["v2"][0]),  # Start trimming
      "-t", str(shortest_duration),
      "-i", video_path2,      #"-to", str(video_durations["v2"][1]),  # End trimming
      #"-filter_complex", f"[1:v]scale={base_dim0}:{base_dim1}[v2]",  # Scale video
      # Third video (bottom left)
      "-ss", str(video_durations["v3"][0]),  # Start trimming
      "-t", str(shortest_duration),
      "-i", video_path3,      #"-to", str(video_durations["v3"][1]),  # End trimming
      #"-filter_complex", f"[2:v]scale={base_dim0}:{base_dim1}[v3]",  # Scale video
      # Merging and positioning videos, combine audio channels
      "-filter_complex",
      "[0:v]scale=960:540[v1]; [1:v]scale=960:540[v2]; [2:v]scale=960:540[v3];[v1][v2][v3]xstack=inputs=3:layout=0_0|w0_0|0_h0[v];[0:a][1:a][2:a]amerge=inputs=3[a]",
      "-map", "[v]",
      "-map", "[a]",
      "-ac", "2",
      "-c:v", "libx264",  # Adjust codec based on your needs
      "-preset", "ultrafast",
      "-r", "5",
      output_path
  ]
  subprocess.run(command, check=True)
