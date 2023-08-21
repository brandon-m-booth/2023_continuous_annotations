# Collecting Annotations with PAGAN
The Platform for Audiovisual General-purpose ANotation (PAGAN) tool was used for crowd-sourcing annotations.  For details about the tool and access to the latest version, see [this paper from Melhart et al.](https://ieeexplore.ieee.org/abstract/document/8925149). An earlier version of the tool was modified to support bounded continuous annotation and used to collect movie violence annotations.  For completeness, that version is provided here along with the following helper scripts.  These scripts were run after PAGAN and its database were set up following its README instructions in the `PAGAN` subfolder.

1. `make_streamable_movie_clips.py`: This script produces streamable movie files (.mp4) from an input csv file containing a list of start/stop cut times for source movies. This was used to clip and prepare the movies for annotation in PAGAN, and it requires *ffmpeg*.  For example, we produced streamable clipped movies for our case study with:
<blockquote>
python make_streamable_movie_clips.py --input_csv movie_cut_times.csv --output_clip_path streamable_movie_clips
</blockquote>

2. `make_pagan_projects.py`: This script anonymizes the names of movie files (mapping readable names to random strings) and outputs a sequence of SQL commands to enter the movies into the PAGAN database. We ran this script on the movie clips produced by the previous script to make them available as annotation tasks in PAGAN. For example, we ran the following command:
<blockquote>
python make_pagan_projects.py --movie_clip_folder streamable_movie_clips --output_anon_movie_folder anon_streamable_movie_clips, --output_sql_file pagan_sql_commands.txt --name_prefix "Movie Violence Rating Experiment"
</blockquote>

From the output `pagan_sql_commands.txt` file, each SQL command was entered into the SQL command line.

# Cleaning PAGAN Annotations
The PAGAN tool streams annotation data in real time over the web to a database for storage.  Since packet losses, connectivity issues, and out-of-order delivery can affect the resulting annotations, we preprocess the annotations to clean these artifacts. The `clean_pagan_logs.py` file does two things: 1) fills in frames with NaN values where "skips" occur in the time series (i.e., missing packets of data), 2) performs a sort to reorder annotation data received out of order. The following command cleans the main study annotations:
<blockquote>
python clean_pagan_logs.py --input_log ../../../dataset/movie_violence/main_study/annotations --output_log ../../../results/main_study/annotations/cleaned
</blockquote>

There is an optional flag `--show_plots` to display the annotations collected in each experiment log.